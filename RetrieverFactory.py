from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm
import time
import ast
import random
seed = 52

class Retriever():
    @abstractmethod
    def fetch_examples(self, query, examples):
        raise NotImplementedError

class RetrieverFactory():
   @classmethod
   def getRetriever(cls, retrieverType, retrieverCfg):
        if retrieverType == "KNN":
           return KNNRetriever(retrieverCfg)
        elif retrieverType == "random":
            return RandomRetriever(retrieverCfg)
        elif retrieverType == "none":
            return ZeroShotRetreiver(retrieverCfg)
        elif retrieverType == "misconception_random":
            return MisconceptionRandomRetriever(retrieverCfg)
        elif retrieverType =="misconception_selection":
            return MisconceptionSelectionRetriever(retrieverCfg)        
        else:
           raise ValueError("Retriever type not supported")
        

class KNNRetriever(Retriever):
    def __init__(self, retrieverCfg):
        self.retrieverCfg = retrieverCfg
        if retrieverCfg.encoderModel:
            self.model_str = retrieverCfg.encoderModel
        else:
            self.model_str = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_str, cache_dir="cache").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str, cache_dir="cache")
        self.max_len, self.emb_size = self.get_max_len_and_emb_size(self.model_str)
        self.embedding_examples = None

    def get_max_len_and_emb_size(self, model_name):
        # https://www.sbert.net/docs/pretrained_models.html
        if "all-MiniLM" in model_name:
            return 256, 384
        if "all-mpnet" in model_name:
            return 384, 768
        if "all-distilroberta" in model_name:
            return 512, 768
        raise ValueError(f"Properties not specified for {model_name}")

    #Mean Pooling - Take attention mask into account for correct averaging
    # Adapted from at https://www.sbert.net/examples/applications/computing-embeddings/README.html
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # q: question only
    # q_a: question and answer
    # q_a_f: question, answer and feedback
    def fetch_examples(self, query, examples: pd.DataFrame):
        # print(query)
        # Parse the questions into the requested example string
        # TODO maybe this should be a factory as well
        if self.retrieverCfg.encodingPattern == "q":
            parsed_examples = examples["question"].tolist()
            # NOTE: I am not sure why i get into error of x is not a string. Bypassing by using str(x)
            parsed_examples = ["question: " + str(x) for x in parsed_examples]
            parsed_query = "question: " + query["question"]
                        
        elif self.retrieverCfg.encodingPattern == "q+a":
            ex_questions = examples["question"].tolist()
            ex_coption = examples["correct_option"].tolist()
            ex_questions = [f"question: {x}" for x in ex_questions]
            ex_correct = [f"correct answer: {x['option']}" for x in ex_coption]
            # concatenate the two lists
            parsed_examples = [q + "\n" + c + "\n" for q, c in zip(ex_questions, ex_correct)]
            # do the same for the query
            q_question = query["question"]
            q_coption = query["correct_option"]
            q_correct = f"correct answer: {q_coption['option']}"
            parsed_query = q_question + "\n" + q_correct
            
        elif self.retrieverCfg.encodingPattern == "q+a+f":
            ex_questions = examples["question"].tolist()
            ex_coption = examples["correct_option"].tolist()
            ex_questions = [f"question: {x}" for x in ex_questions]
            ex_explanation = [f"correct explanation: {x['explanation']}" for x in ex_coption]
            ex_correct = [f"correct answer: {x['option']}" for x in ex_coption]
            # concatenate the three lists
            parsed_examples = [q + "\n" + e + "\n" + c for q, c, e in zip(ex_questions, ex_correct, ex_explanation)]
            
            # do the same for the query
            q_question = query["question"]
            q_coption = query["correct_option"]
            q_correct = f"correct answer: {q_coption['option']}"
            q_explanation = f"correct explanation: {q_coption['explanation']}"
            parsed_query = q_question + "\n" + q_explanation + "\n" + q_correct
            
        else:
            raise ValueError("Encoding pattern not supported")

        if self.embedding_examples is None:
            # Compute SBERT encoding of each example
            # start_time = time.time()
            with torch.no_grad():
                # TODO move me as this program grows, maybe utilities
                dataloader = DataLoader(parsed_examples, batch_size=self.retrieverCfg.batch_size)
                all_embeddings = []
                for _, batch_inputs in enumerate(tqdm(dataloader)):
                    token_batch = self.tokenizer(batch_inputs, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt').to(self.device)
                    # Call the forward pass of BERT on the batch inputs and get mean pooled embeddings
                    batch_outputs = self.model(input_ids=token_batch['input_ids'], attention_mask=token_batch['attention_mask'])
                    all_embeddings.append(self.mean_pooling(batch_outputs["last_hidden_state"], token_batch['attention_mask']))
                embedding_examples = torch.cat(all_embeddings, dim=0)
                # Normalize the embedding
                # embedding_examples = embedding_examples / embedding_examples.norm(dim=1, keepdim=True)
            # end_time = time.time()
            # print(f"SBERT encoding elapsed: {end_time - start_time}")
            self.embedding_examples = embedding_examples
        else:
            embedding_examples = self.embedding_examples

        # Compute SBERT encoding of query
        # TODO This does not need to be recomputed every time as it is already for the examples
        token_query = self.tokenizer(parsed_query, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt').to(self.device) # (examples, seq, hidden)
        with torch.no_grad():
            model_output_query = self.model(**token_query)
            embedding_query = self.mean_pooling(model_output_query["last_hidden_state"], token_query["attention_mask"]) # (examples, embedding)
            # embedding_query = embedding_query / embedding_query.norm(dim=1, keepdim=True)

        # TODO make this a hyperparameter
        # Compute cosine similarity between query and each example
        
        start_time = time.time()
        # TODO With a normalized embedding, this is the same as dot product
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos_sim(embedding_query, embedding_examples) # (examples)
        sorted_examples = [examples.iloc[i.item()] for i in np.argsort(-cos_sim.cpu())] # negative for descending order
        # convert back to pandas df for easier manipulation
        sorted_examples_df = pd.concat(sorted_examples, axis=1).T.reset_index(drop=True)
        if self.retrieverCfg.exclude_construct_level:
            # Define a custom function to extract the value from the dictionary
            construct_str = f"construct{self.retrieverCfg.exclude_construct_level}"
            def check_value(row, query, construct_str):
                return row['construct_info'][construct_str][0] != query["construct_info"][construct_str][0]
            # Use the .apply() method to apply the custom function to each row
            sorted_examples_df = sorted_examples_df[sorted_examples_df.apply(check_value, args=(query, construct_str,), axis=1)]
        end_time = time.time()
        # print(f"similarity computing elapsed: {end_time - start_time}")

        # In case  you wanna try L2
        # l2_sim = torch.sqrt(torch.sum(torch.square(embedding_query - embedding_examples), dim=1))
        # sorted_examples = [examples.iloc[i.item()] for i in np.argsort(l2_sim)] # ascending order

        # Return the n closest examples to the query
        assert self.retrieverCfg.k < len(examples), "k must be less than the number of examples"
        # NOTE: We assume the first example is the query itself if it's in the pool of examples
        if sorted_examples_df.iloc[0].equals(query):
            top_k = sorted_examples_df.iloc[1:self.retrieverCfg.k+1]
        else:
            top_k = sorted_examples_df.iloc[:self.retrieverCfg.k]
        return top_k

class RandomRetriever(Retriever):
    def __init__(self, retrieverCfg):
        self.retrieverCfg = retrieverCfg
        random.seed(seed)

    def fetch_examples(self, query, examples: pd.DataFrame):
        # TODO we need this to work for same construct
        # Get random set of examples
        samples = examples.sample(self.retrieverCfg.k + 1, random_state=5)
        # Ensure that query sample is not included in set
        if "option" in query:
            query_idx = (samples["question"] == query["question"]) & (samples["option"] == query["option"])
        else:
            query_idx = samples["question"] == query["question"]
        samples = samples[~query_idx]
        return samples.iloc[:self.retrieverCfg.k]
    
class MisconceptionRandomRetriever(Retriever):
    def __init__(self, retrieverCfg):
        self.retrieverCfg = retrieverCfg
        random.seed(seed)

    def fetch_examples(self, query, examples: pd.DataFrame):
        # TODO we need this to work for same construct
        # Get random set of examples
        examples = ast.literal_eval(query['misconceptions'])
        if len(examples) > 2:
            samples = random.sample(examples, self.retrieverCfg.k)
        else:
            samples = examples
        return samples

class MisconceptionSelectionRetriever(Retriever):
    def __init__(self, retrieverCfg):
        self.retrieverCfg = retrieverCfg

    def fetch_examples(self, query, examples: pd.DataFrame):
        # TODO we need this to work for same construct
        # Get random set of examples
        examples = ast.literal_eval(query['misconceptions'])
        return examples

class ZeroShotRetreiver(Retriever):
    def __init__(self, retrieverCfg):
        self.retrieverCfg = retrieverCfg
    
    def fetch_examples(self, query, examples: pd.DataFrame):
        # This is just a compatibility thing
        return [] 

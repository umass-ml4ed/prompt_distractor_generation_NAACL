from omegaconf import DictConfig, OmegaConf
from PromptFactory import PromptFactory as pf
from OpenAIInterface import OpenAIInterface as oAI
from RetrieverFactory import RetrieverFactory as rf
import hydra
import pandas as pd
from utils import save_result, str_to_dict_eedi_df
import time
from tqdm import tqdm
import atexit
import json

def evalautePrompts(df_pool, df_train, cfg):
    ## This function is the crux of the experiment
    retriver = rf.getRetriever(cfg.retriever.type, cfg.retriever)
    
    # TODO there's probably a sleeker way to batch this
    results = []
    prompts = []
    all_in_context_examples = [] # will be a list of lists
    print("Generating prompts...")
    gen_tic = time.time()
    if cfg.development.truncate:
        df_train = df_train.iloc[0:cfg.development.truncate]
    for i, sample in tqdm(df_train.iterrows(), total=len(df_train)):
        in_context_examples = retriver.fetch_examples(sample, df_pool)
        all_in_context_examples.append(in_context_examples)
        # print("in_context_examples", in_context_examples)
        prompt = pf.producePrompt(sample, cfg.prompt.type, examples=in_context_examples)
        # print("prompt", prompt)
        prompts.append(prompt)
    gen_toc = time.time()
    print("Generated", len(prompts), "prompts in", gen_toc - gen_tic, "seconds.")
    # NOTE: this currently blocks the main thread, it would be more efficient to encode in batches
    # And then send the batches to the API
        
    if cfg.dir_finetune_result.model_name == "mistral_finetune" or cfg.dir_finetune_result.model_name == "gpt_finetune" or cfg.dir_finetune_result.model_name == "SB_sampling":
        print("Loading predictions from fine-tuned model...")
        with open(str(cfg.dir_finetune_result.model_name) + ".json", "r") as f:
            prompt_responses = json.load(f)
    else:
        print("Calling OpenAI API...")
        prompt_tic = time.time()
        if cfg.openAI.model in oAI.CHAT_GPT_MODEL_NAME:
            prompt_responses = oAI.getCompletionForAllPrompts(cfg.openAI, prompts, batch_size=20, use_parallel=True)
        else:
            prompt_responses = oAI.getCompletionForAllPrompts(cfg.openAI, prompts, batch_size=10, use_parallel=False)
        prompt_toc = time.time()
        print("Called OpenAI API in", prompt_toc - prompt_tic, "seconds.")
        
    print("Processing responses...")
    process_tic = time.time()
    for prompt, response in zip(prompts, prompt_responses):
        if cfg.dir_finetune_result.model_name == "llama_finetune" or cfg.dir_finetune_result.model_name == "gpt_finetune" or cfg.dir_finetune_result.model_name == "gpt_sampling":
            pred = response
        else:
            pred = response["text"] if "davinci" in cfg.openAI.model else response["message"]["content"]
        result = {"prompt": prompt, "raw_response": pred}
        results.append(result)

    ## Save output to file
    results_df = pd.DataFrame(results)
    save_result(results_df, cfg)
    process_toc = time.time()
    print("Processed", len(results), "responses in", process_toc - process_tic, "seconds.")
    
    

@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    print("Config Dump:\n" + OmegaConf.to_yaml(cfg))

    # Save openAI response cache at program exit
    atexit.register(oAI.save_cache)

    if cfg.command.task == "open_ai_fetch":
        df_pool = pd.read_csv(cfg.data.trainFilepath)
        df_test = pd.read_csv(cfg.data.testFilepath)
        df_pool = str_to_dict_eedi_df(df_pool)
        df_test = str_to_dict_eedi_df(df_test)
        evalautePrompts(df_pool, df_test, cfg)
    else:
        print("Unrecognized task:", cfg.command.task)

if __name__ == "__main__":
    main()

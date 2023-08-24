import torch
# import pandas
import pickle
import gc
import argparse

# import datasets
import transformers
# from datasets import Dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig

# import torch_geometric.transforms as T
# from ogb.nodeproppred import PygNodePropPredDataset

from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
# from accelerate.utils import get_balanced_memory

from utils import *


def llm_query (total_prompts=[], model_name="", problem_name=""):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    print('token model loaded')
    llm_response_collections = []
    
    print('start query')
    for idx in range(0, 50):    
        pp_prompt = total_prompts[idx]

        inputs = tokenizer(pp_prompt, truncation=True, max_length=4096, return_tensors="pt").input_ids
        inputs = inputs.to('cuda')
        outputs = model.generate(inputs, max_length=4096)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # print(idx)
        # print(output_text)
        # print('====================================')

        outputs.cpu()
        llm_response_collections.append([idx, outputs, output_text])

        del(inputs)
        del(outputs)
        del(output_text)
        del(pp_prompt)
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(llm_response_collections, f"./llm_response_collections_{problem_name}.pt")
    print(f'results saved')
    return llm_response_collections

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", type=str, choices=['simple_path', 'node_neighbor_1hop', 'node_neighbor_2hop', 'node_neighbor_3hop', 'find_wedge_centering', 'find_wedge_noncentering', 'find_triangle', 'node_degree', 'shortest_path', 'shortest_path_verify'],
                        help="specify problem type")
    parser.add_argument("--prompt_type", type=str, choices=['standard', 'cot', 'fewshot'],
                        help="specify prompt type")
    parser.add_argument("--graph_type", type=str, default='undirected', choices=['undirected', 'directed'],
                        help="specify graph type") 
    parser.add_argument("--model_name", type=str, default='gpt3turbo', choices=['Vicuna', 'Lazarus', 'gpt3turbo', 'gpt4'],
                        help="specify model name")   
    args = parser.parse_args()


    # model selection
    if args.model_name == 'Vicuna':
        model_name = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    elif args.model_name == 'Lazarus':
        model_name = "CalderaAI/30B-Lazarus"
    else:
        print(f'##### YOU ARE USING CHATGPT MODEL, PLEASE NOTE THAT YOU NEED TO MANNUALLY COLLECT ALL RESPONSES FROM CHATGPT! #####')
        print(f'##### THIS MODEL WILL NEED AN OPENAI ACCOUNT AND MAY GENERATE ADDITIONAL FEES FROM OPENAI! #####')

    # prompt generation
    medium_graph_base = pickle.load(open('./medium_undir_graph_50.pickle', 'rb'))
    
    graph_count = len(medium_graph_base)
    total_prompts = []
    addition_num = 0
    source = 0
    target = 0
    node_list = []
    this_examplar = ''
    total_node_list = []
    node_degree_all_list = []
    first_round_ans_collections = []

    if args.problem_name == 'simple_path':
        node_pairs = torch.load('./node_pairs_medium_50graphs.pt')
    elif args.problem_name == 'shortest_path':
        node_pairs = torch.load('./node_pairs_medium_50graphs.pt')
    elif args.problem_name == 'shortest_path_verify':
        node_pairs = torch.load('./node_pairs_medium_50graphs.pt')
        first_round_ans_collections = torch.load('./all_shortest_path_first_round.pt')
    elif args.problem_name == 'node_neighbor_1hop':
        total_node_list = torch.load('./node_list_1hop_nodeneighbor.pt')
        addition_num = 1
    elif args.problem_name == 'node_neighbor_2hop':
        total_node_list = torch.load('./node_list_2hop_nodeneighbor.pt')
        addition_num = 2        
    elif args.problem_name == 'node_neighbor_3hop':
        total_node_list = torch.load('./node_list_3hop_nodeneighbor.pt')
        addition_num = 3
    elif (args.problem_name == 'find_wedge_centering' or args.problem_name == 'find_wedge_noncentering'):
        wedge_selections = torch.load('./node_list_findwedge.pt')
    elif args.problem_name == 'find_triangle':
        triangle_selections = torch.load('./node_list_with_triangles_findtriangle.pt')
    elif args.problem_name == 'node_degree':
        pass
    else:
        print("ERROR: PROBLEM TYPE NOT DEFINED!")
        return False

    if args.prompt_type == 'fewshot':
        if args.model_name == 'gpt4':
            fewshot_examplars = torch.load('./all_examplar_prompts_for_fewshotprompt_gpt4.pt')
        else:
            fewshot_examplars = torch.load('./all_examplar_prompts_for_fewshotprompt.pt')
        this_examplar = fewshot_examplars[args.problem_name]

    # generate prompt one by one
    for idx in range(0, graph_count):
        test_graph = medium_graph_base[idx]

        if args.problem_name == 'simple_path' or args.problem_name == 'shortest_path':
            source, target = node_pairs[idx]
        elif (args.problem_name == 'node_neighbor_1hop' or args.problem_name == 'node_neighbor_2hop' or args.problem_name == 'node_neighbor_3hop'):
            target, node_list = total_node_list[idx]
        elif (args.problem_name == 'find_wedge_centering' or args.problem_name == 'find_wedge_noncentering'):
            target = wedge_selections[idx]
        elif args.problem_name == 'find_triangle':
            target, correct_answer = triangle_selections[idx]
        elif args.problem_name == 'node_degree':
            while len(node_list) < 5:
                rand_num = random.randint(0, (len(test_graph.nodes) - 1))
                if (not (rand_num in node_list)):
                    node_list.append(rand_num)
            node_list.sort()

        if len(first_round_ans_collections) > 0:
            first_round_ans = first_round_ans_collections[idx]
        else:
            first_round_ans = []

        test_prompt = prompt_standard_generator(test_graph, source=source, target=target, problem_name=args.problem_name, graph_type=args.graph_type, prompt_type=args.prompt_type, prompt_examplar=this_examplar, addition_num=addition_num, node_list=node_list, first_round_ans=first_round_ans)

        if args.problem_name == 'node_degree':
            node_degree_all_list.append(node_list)
            total_prompts.append(test_prompt)
        else:
            total_prompts.append(test_prompt)


    if (args.model_name == 'Vicuna' or args.model_name == 'Lazarus'):
        llm_response_collections = llm_query(total_prompts, model_name, args.problem_name)
    else:
        print(f'PLEASE USE THE CORRECT CHATGPT MODEL YOU WANT ON CHATGPT WEBSITE')
        print(f'COPY THE FOLLOWING SENTENCES TO THE CHATGPT PAGE AND COPY THE RESPONSE BACK')
        llm_response_collections = []
        for e_idx in range(0, graph_count):
            print(f'HERE IS THE NUMBER {e_idx} QUERY: COPY MESSAGE BELOW TO CHATGPT')
            print(total_prompts[e_idx])
            raw_response = input('COPY RESPONSE FROM CHATGPT HERE : --->   ')
            llm_response_collections.append(raw_response)

    print(f"PLEASE MANNUALLY CHECK ALL RESPONSES AND ENSURE THE RESULTS ARE PROPERLY LISTED AFTER THE TERM 'ANS:'")
    llm_refined_answer = llm_response_collections

    correct_ratio = get_final_results(llm_refined_answer=llm_refined_answer, problem_name=args.problem_name, total_node_list=total_node_list, wedge_selections=wedge_selections, triangle_selections=triangle_selections, node_degree_all_list=node_degree_all_list, first_round_asp=first_round_ans_collections,)

    print(f'correct_ratio = {correct_ratio}')

    return True

if __name__ == "__main__":
    print(f'start executing main func')
    main()
    print(f'finish executing main func')
import torch
import pickle
import random
import networkx

medium_graph_base = pickle.load(open('./medium_undir_graph_50.pickle', 'rb'))
node_pairs = torch.load('./node_pairs_medium_50graphs.pt')


# support funcs for generating prompts

def prompt_standard_generator(graph, source=0, target=0, problem_name='all_shortest_path', graph_type='undirected', prompt_type='standard', prompt_examplar='', addition_num=0, node_list=[], first_round_ans=[]):
    if graph_type == 'undirected':
        pp_head = 'Q: Given the undirected graph with the specified nodes and edges,'
    else:
        pp_head = 'Q: Given the directed graph with the specified nodes and edges,'
        
    pp_node_list = ' nodes: ' + str(graph.nodes) + ','
    pp_edge_list = ' edges: ' + str(graph.edges) + ','
    pp_tail = ""
    if problem_name == 'shortest_path':
        pp_tail = f' find all the correct shortest paths from node {source} to node {target}, list all the answers after "Ans:" in the format of [0-1-2] and separate the answers by a comma.'
    elif problem_name == 'shortest_path_verify':
        pp_tail = f' the following are some potential shortest paths from node {source} to node {target}, choose all the correct ones from the list of possible shortest paths provided, list all the answers after "Ans:" in the format of [0-1-2] and separate the answers by a comma. The possible shortest paths are: '    
        for p_id in range(0, len(first_round_ans)):
            if p_id != (len(first_round_ans) - 1):
                pp_tail = pp_tail + first_round_ans[p_id] + ', '
            else:
                pp_tail = pp_tail + first_round_ans[p_id] + '.'
    elif problem_name == 'count_triangles':
        pp_tail = f' find all the triangles connected by three edges in the given graph, list all the triangles after "Ans:" in the format of [0-1-2] and separate the answers by a comma.'
    elif problem_name == 'find_random_triangle':
        pp_tail = f' find a single triangle connected by three edges in the given graph, list the triangle after "Ans:" in the format of [0-1-2].'
    elif problem_name == 'find_triangle':
        pp_tail = f' find a single triangle containing node {target} in the given graph, the triangle must be connected by three edges, list the triangle after "Ans:" in the format of [0-1-2].'
    elif problem_name == 'find_wedge_centering':
        pp_tail = f' find a single wedge containing node {target} in the given graph, node {target} must be the center of this wedge, list the wedge after "Ans:" in the format of [0-1-2].'
    elif problem_name == 'find_wedge_noncentering':
        pp_tail = f' find a single wedge containing node {target} in the given graph, list the only one wedge after "Ans:" in the format of [0-1-2].'
    elif problem_name == 'find_cliques':
        pp_tail = f' find all the cliques with {addition_num} nodes in the given graph, list all the cliques after "Ans:" in the format of [0-1-2] and separate the answers by a comma.'
    elif problem_name == 'graph_bisection':
        pp_tail = f' partition the graph into two parts with roughly similar number of nodes, minimize the number of edge cuts across the two parts, no overlapping nodes are allowed, list the two parts after "Ans:" in the format of [0-1-2] and separate them by a comma.'
    elif problem_name == 'simple_path':
        pp_tail = f' find a single path from node {source} to node {target} connected by edges in the given graph, list the answer after "Ans:" in the format of [0-1-2].'
    elif problem_name == 'single_shortest_path':
        pp_tail = f' find a single shortest path from node {source} to node {target} connected by edges in the given graph, list the answer after "Ans:" in the format of [0-1-2].'
    elif (problem_name == 'node_neighbor_1hop' or problem_name == 'node_neighbor_2hop' or problem_name == 'node_neighbor_3hop'):
        pp_tail = f' Are nodes '
        node_list.sort()
        for ch in node_list:
            pp_tail = pp_tail + str(ch)
            if ch != node_list[-1]:
                pp_tail = pp_tail + ', '
            else:
                pp_tail = pp_tail + ' ' 
        pp_tail = pp_tail + f'the {addition_num}-hop neighbors of node {target}? List the answers after "Ans:" in the format of [Yes, No,].'
    elif problem_name == 'node_degree':
        pp_tail = f' find the node degree of node '
        for ch in node_list:
            pp_tail = pp_tail + str(ch)
            if ch != node_list[-1]:
                pp_tail = pp_tail + ', '
            else:
                pp_tail = pp_tail + ' ' 
        pp_tail = pp_tail + f'in the given graph, list the answer after "Ans:". in the format of [1, 2, 3,].'
    elif problem_name == 'node_eccentricity':
        pp_tail = f' find the eccentricity of node {addition_num} in the given graph, list the answer after "Ans:".'
    elif problem_name == 'edge_cover':
        pp_tail = f' find a set of edges that can form an edge cover in the given graph, list the answer after "Ans:" in the format of [(0, 1),].'
    else:
        print(f"problem name {problem_name} not defined!")
        return None

    if prompt_type == 'standard':
        pp_prompt = pp_head + pp_node_list + pp_edge_list + pp_tail
    elif prompt_type == 'cot':
        pp_prompt = pp_head + pp_node_list + pp_edge_list + pp_tail + " A: Let's think step by step, "
    elif prompt_type == 'fewshot':
        pp_prompt = prompt_examplar + ' ' + pp_head + pp_node_list + pp_edge_list + pp_tail
    else:
        print(f"prompt type {prompt_type} not defined!")
        return None

    return pp_prompt


### support funcs for find triagnle problem
def permute_all_triangles (all_triangles):
    permuted_collect_triangles = []
    for elem in all_triangles:
        perm_elem = list(itertools.permutations(elem))
        for p_elem in perm_elem:
            temp = ""
            for ch in p_elem:
                temp = temp + str(ch)
                if ch != p_elem[-1]:
                    temp = temp + '-'
            permuted_collect_triangles.append(temp)
    return permuted_collect_triangles


### support funcs for wedge counting problem
def compare_results_find_wedge (graph, target_node, response_answer, position='center'):
    
    result_index = 0

    for ans in response_answer:
        ans_nums = []
        idx_rec = 0
        single_index = 1
        for i in range(0, len(ans)):
            if i < idx_rec:
                continue

            temp = ''

            if ans[i].isdigit():
                temp = temp + ans[i]
                if i == (len(ans) - 1):
                    ans_nums.append(int(temp))
                    break

                for j in range((i + 1), len(ans)):
                    if ans[j].isdigit():
                        temp = temp + ans[j]
                        if j == (len(ans) - 1):
                            ans_nums.append(int(temp))
                            idx_rec = j + 1
                            break
                    else:
                        idx_rec = j
                        ans_nums.append(int(temp))
                        break

        # print(ans)
        # print(ans_nums)

        if (not target_node in ans_nums):
            print("Target Node Not Found!")
            return 0
        
        if position == 'center':
            for n_idx in ans_nums:
                if n_idx < target_node:
                    edg = (n_idx, target_node)
                    if (not edg in graph.edges):
                        single_index = 0
                elif n_idx > target_node:
                    edg = (target_node, n_idx)
                    if (not edg in graph.edges):
                        single_index = 0
        elif position == 'noncenter':
            edge_count = 0
            ans_nums.sort()
            edge_list = [(ans_nums[0], ans_nums[1]), (ans_nums[0], ans_nums[2]), (ans_nums[1], ans_nums[2])]
            for edg in edge_list:
                if edg in graph.edges:
                    edge_count += 1

            if edge_count < 2:
                single_index = 0

        if single_index == 1:
            result_index += 1

    # ensure every answer in response is correct, the whole answer is wrong if any answer is wrong, since only one answer should be generated
    if result_index >= len(response_answer):
        return result_index

    return 0


def get_numeric_results_find_wedge (pp_response):

    ans_index = pp_response.find('Ans:') + 4
    answer_collections = []
    rec_index = 0

    for i in range(ans_index, len(pp_response)):
        if i < rec_index:
            continue

        temp = ""
        if pp_response[i].isdigit():
            temp = temp + pp_response[i]
            ### note!!! when i == len(pp_response) - 1, the func returns without append temp
            for j in range((i + 1), len(pp_response)):
                if pp_response[j].isdigit():
                    temp = temp + pp_response[j]

                    if j == (len(pp_response) - 1):
                        answer_collections.append(temp)
                        rec_index = j
                        break
                else:
                    if pp_response[j] == ',' or pp_response[j] == ']' or (j == (len(pp_response) - 1)):
                        if temp[-1] == '-':
                            temp = temp[:-1]
                        answer_collections.append(temp)
                        rec_index = j
                        break
                    
                    if temp[-1] != '-':
                        temp = temp + '-'

    # print(answer_collections)
    # print(len(answer_collections))
    return answer_collections


### support funcs for node neighbor problems
def get_node_neighbors (graph, target_node=-1):

    if target_node == -1:
        target_node = random.randint(0, (len(graph.nodes) - 1))
    neighbor_1hop = list(graph.neighbors(target_node))
    neighbor_2hop = []
    neighbor_3hop = []
    prev_node_list = []
    other_node_list = []
    prev_node_list.append(target_node)
    
    for nidx in neighbor_1hop:
        prev_node_list.append(nidx)

    for ch in neighbor_1hop:
        temp_nb = list(graph.neighbors(ch))
        for elem in temp_nb:
            if (not elem in prev_node_list):
                neighbor_2hop.append(elem)
                prev_node_list.append(elem)

    for ch in neighbor_2hop:
        temp_nb = list(graph.neighbors(ch))
        for elem in temp_nb:
            if (not elem in prev_node_list):
                neighbor_3hop.append(elem)
                prev_node_list.append(elem)            

    for ch in graph.nodes:
        if (not ch in prev_node_list):
            other_node_list.append(ch)

    return target_node, neighbor_1hop, neighbor_2hop, neighbor_3hop, other_node_list


def clean_classify_results (pp_response):
    pp_clean_answer = []
    ans_index = pp_response.find('Ans:') + 4
    for i_idx in range(ans_index, (len(pp_response) - 1)):
        y_temp = ""
        if i_idx < (len(pp_response) - 2):
            y_temp = pp_response[i_idx] + pp_response[i_idx + 1] + pp_response[i_idx + 2]
        n_temp = pp_response[i_idx] + pp_response[i_idx + 1]
        if y_temp == 'Yes':
            pp_clean_answer.append(int(1))
        elif n_temp == 'No':
            pp_clean_answer.append(int(0))

    return pp_clean_answer


#### support funcs for simple path problem

def compare_results_simple_path(answer_collections, graph):

    final_results = []
    edg = list(graph.edges)
    for ans in answer_collections:
        ans_nums = []
        idx_rec = 0

        for i in range(0, len(ans)):
            if i < idx_rec:
                continue

            temp = ''

            if ans[i].isdigit():
                temp = temp + ans[i]
                if i == (len(ans) - 1):
                    ans_nums.append(int(temp))
                    break

                for j in range((i + 1), len(ans)):
                    if ans[j].isdigit():
                        temp = temp + ans[j]
                        if j == (len(ans) - 1):
                            ans_nums.append(int(temp))
                            idx_rec = j + 1
                            break
                    else:
                        idx_rec = j
                        ans_nums.append(int(temp))
                        break
        
        # print(ans)
        # print(ans_nums)
        
        correctness_indicator = 1
        for e_idx in range(0, len(ans_nums)):
            if e_idx < (len(ans_nums) - 1):
                if ans_nums[e_idx] < ans_nums[e_idx + 1]:
                    query_edge = (ans_nums[e_idx], ans_nums[e_idx + 1])
                else:
                    query_edge = (ans_nums[e_idx + 1], ans_nums[e_idx])
                if (not query_edge in edg):
                    correctness_indicator = 0
                    break
        
        if correctness_indicator == 0:
            final_results.append(0)
        else:
            final_results.append(1)

    return final_results

# obtain the list of all shortest paths in a graph
def all_shortest_paths (graph, source_id, target_id):
    paths = networkx.all_shortest_paths(graph, source=source_id, target=target_id)
    collect_paths = []
    for elem in paths:
        temp = ""
        for ch in elem:
            temp = temp + str(ch)
            if ch != elem[-1]:
                temp = temp + '-'
        collect_paths.append(temp)
    return collect_paths

def compare_results(answer_collections, correct_answers):
    hit_count = 0
    fal_count = 0
    for ans in answer_collections:
        if ans in correct_answers:
            hit_count += 1
        else:
            fal_count += 1

    return hit_count, fal_count

# get the answers from gpt for node degree problem
def get_numeric_results_node_degree (pp_response):

    ans_index = pp_response.find('Ans:') + 4
    answer_collections = []
    rec_index = 0

    for i in range(ans_index, len(pp_response)):
        if i < rec_index:
            continue

        temp = ""
        if pp_response[i].isdigit():
            temp = temp + pp_response[i]
            
            for j in range((i + 1), len(pp_response)):
                if pp_response[j].isdigit():
                    temp = temp + pp_response[j]

                    if j == (len(pp_response) - 1):
                        answer_collections.append(int(temp))
                        rec_index = j
                        break
                else:
                    rec_index = j
                    answer_collections.append(int(temp))
                    break

    # print(answer_collections)
    # print(len(answer_collections))
    return answer_collections


def get_numeric_results (pp_response):

    ans_index = pp_response.find('Ans:') + 4
    answer_collections = []
    rec_index = 0

    for i in range(ans_index, len(pp_response)):
        if i < rec_index:
            continue

        temp = ""
        if pp_response[i].isdigit():
            temp = temp + pp_response[i]
            ### note!!! when i == len(pp_response) - 1, the func returns without append temp
            for j in range((i + 1), len(pp_response)):
                if pp_response[j].isdigit():
                    temp = temp + pp_response[j]

                    if j == (len(pp_response) - 1):
                        answer_collections.append(temp)
                        rec_index = j
                        break
                else:
                    if pp_response[j] == ',' or (j == (len(pp_response) - 1)):
                        if temp[-1] == '-':
                            temp = temp[:-1]
                        answer_collections.append(temp)
                        rec_index = j
                        break
                    
                    if temp[-1] != '-':
                        temp = temp + '-'

    # print(answer_collections)
    # print(len(answer_collections))
    return answer_collections


### need to mannually check shortest path / all shortest path, see if the source/taget nodes are correct or not
def get_final_results(llm_refined_answer=[], problem_name='', total_node_list=[], wedge_selections=[], triangle_selections=[], node_degree_all_list=[], first_round_asp=[]) :

    if problem_name == 'simple_path':
        for e_idx in range(0, len(llm_refined_answer)):
            if type(llm_refined_answer[e_idx]) != str:
                non_related_response += 1
                continue

            if llm_refined_answer[e_idx].find("Ans:") < 0:
                non_right_response += 1
                print(f'ERROR: NUMBER {e_idx} RESPONSE NOT IN CORRECT FORMAT!')
            
            test_graph = medium_graph_base[e_idx]
            test_clean_answer = get_numeric_results(llm_refined_answer[e_idx])
            # print(test_clean_answer)
            res = compare_results_simple_path(test_clean_answer, test_graph)
            # print(res)
            if res[0] > 0:
                print(f'*** this is correct ***')
                hit_count += 1
            # break

            # print(f'ans at {e_idx}')
            # print(f'node pairs {node_pairs[e_idx]}')
            # print(f'clean answer {test_clean_answer}')
            # print('==============')
        
        # print(hit_count)
        correct_ratio = hit_count / 50
    
    elif (problem_name == 'node_neighbor_1hop' or problem_name == 'node_neighbor_2hop' or problem_name == 'node_neighbor_3hop'):

        correct_count = 0
        fal_count = 0
        non_response_count = 0

        for elem in range(0, len(llm_refined_answer)):
            # print(elem)
            node_1, node_list = total_node_list[elem]
            test_response = llm_refined_answer[elem]

            if type(test_response) != str:
                non_response_count += 1
                continue

            if test_response.find("Ans:") < 0:
                print(f'ERROR: NUMBER {elem} RESPONSE NOT IN CORRECT FORMAT!')

            pp_clean_result = clean_classify_results(test_response)
            # print(pp_clean_result)
            test_graph = medium_graph_base[elem]
            node_1, neighbor_1, neighbor_2, neighbor_3, other_nodes = get_node_neighbors(test_graph, target_node=node_1)
            node_list.sort()
            
            min_length = len(pp_clean_result) if (len(node_list) > len(pp_clean_result)) else len(node_list)
            if problem_name == 'node_neighbor_1hop':
                target_neighbor = neighbor_1
            elif problem_name == 'node_neighbor_2hop':
                target_neighbor = neighbor_2
            else:
                target_neighbor = neighbor_3

            for ch in range(0, min_length):
                correct_index = 0
                if node_list[ch] in target_neighbor:
                    correct_index = 1
                
                if pp_clean_result[ch] == correct_index:
                    correct_count += 1
                else:
                    fal_count += 1
                    # print(f'{elem} node {node_1} neighbor {node_list[ch]} decision {pp_clean_result[ch]} <-------->  correct {correct_index}')

        # print(non_response_count)
        # print(correct_count)
        # print(correct_count / 250)
        # print(fal_count)
        # print(fal_count / 250)
        correct_ratio = correct_count / 250
    
    elif (problem_name == 'find_wedge_centering' or problem_name == 'find_wedge_noncentering'):

        correct_count = 0
        fal_count = 0
        non_response_count = 0

        for e_idx in range(0, 50):
            target_node = wedge_selections[e_idx]
            test_graph = medium_graph_base[e_idx]

            if type(llm_refined_answer[e_idx]) != str:
                non_response_count += 1
                continue
            
            if llm_refined_answer[e_idx].find("Ans:") < 0:
                print(f'ERROR: NUMBER {e_idx} RESPONSE NOT IN CORRECT FORMAT!')

            response_answer = get_numeric_results_find_wedge(llm_refined_answer[e_idx])

            if problem_name == 'find_wedge_centering':
                single_index = compare_results_find_wedge (test_graph, target_node, response_answer)
            elif problem_name == 'find_wedge_noncentering':
                single_index = compare_results_find_wedge (test_graph, target_node, response_answer, position='noncenter')
        
            if single_index == 0:
                fal_count += 1
                # print(target_node)
                # print(response_answer)
                # print(test_graph.edges)
                # print(f'=================')
            else:
                correct_count += 1
                # print(f'*** correct ***')
                # print(target_node)
                # print(response_answer)
                # print(test_graph.edges)
                # print(f'=================')
                    
        # print(non_response_count)
        # print(correct_count)
        # print(correct_count / (50 - non_response_count))
        # print(correct_count / 50)
        # print(fal_count)
        # print(fal_count / 50)
        
        correct_ratio = correct_count / 50

    elif problem_name == 'find_triangle':
        correct_count = 0
        fal_count = 0
        non_response_count = 0
        for e_idx in range(0, 50):
            target_node, correct_answer = triangle_selections[e_idx]
            if type(llm_refined_answer[e_idx]) != str:
                non_response_count += 1
                continue

            if llm_refined_answer[e_idx].find("Ans:") < 0:
                print(f'ERROR: NUMBER {e_idx} RESPONSE NOT IN CORRECT FORMAT!')
                break

            response_answer = get_numeric_results(llm_refined_answer[e_idx])

            permuted_correct_answer = permute_all_triangles(correct_answer)

            single_index = 0
            for elem in response_answer:
                if elem in permuted_correct_answer:
                    single_index += 1
            
            # ensure every answer in response is correct, the whole answer is wrong if any answer is wrong, since only one answer should be generated
            if single_index < len(response_answer):
                fal_count += 1
                # print(f'target node -> {target_node}')
                # print(response_answer)
                # print(correct_answer)
                # print(f'=================')
            else:
                correct_count += 1
                # print(f'*** correct answer ***')
                # print(f'target node -> {target_node}')
                # print(response_answer)
                # print(correct_answer)
                # print(f'=================')

        # print(non_response_count)
        # print(correct_count)
        # print(correct_count / (50 - non_response_count))
        # print(correct_count / 50)
        # print(fal_count)
        # print(fal_count / 50)

        correct_ratio = correct_count / 50

    elif problem_name == 'node_degree':
        correct_count = 0
        fal_count = 0
        non_response_count = 0

        for e_idx in range(0, 50):
            node_list = node_degree_all_list[e_idx]
            test_response = llm_refined_answer[e_idx]
            if type(test_response) != str:
                non_response_count += 1
                continue

            if test_response.find("Ans:") < 0:
                print(f'ERROR: NUMBER {e_idx} RESPONSE NOT IN CORRECT FORMAT!')
                break

            # node_list, pp_prompt = total_prompts[e_idx]
            # print(node_list)
            test_graph = medium_graph_base[e_idx]
            test_clean_answer = get_numeric_results_node_degree(test_response)

            min_length = len(node_list) if len(node_list) <= len(test_clean_answer) else len(test_clean_answer)
                
            for n_idx in range(0, min_length):
                n_id = node_list[n_idx]
                correct_answer = test_graph.degree(n_id)

                if correct_answer == test_clean_answer[n_idx]:
                    correct_count += 1
                else:
                    fal_count += 1
                    # print(f'{e_idx} -- {n_idx} : correct answer {correct_answer} <======> gpt answer {test_clean_answer[n_idx]}')
                    # print(f'===================================')

        correct_ratio = correct_count / (50 * 5)
        # fal_ratio = fal_count / (50 *5)

        # print(non_response_count)
        # print(correct_count)
        # print(correct_count / (250 - non_response_count * 5 - 39))
        # print(correct_ratio)
        # print(fal_count)

    
    elif (problem_name == 'shortest_path' or problem_name == 'shortest_path_verify'):
        res_collections = []
        first_round_asp = []
        for e_idx in range(0, 50):
            source_node, target_node = node_pairs[e_idx]
            this_graph = medium_graph_base[e_idx]
            correct_paths = all_shortest_paths(this_graph, source_node, target_node)

            this_response = llm_refined_answer[e_idx]

            if type(this_response) != str:
                hit_count = 0
                fal_count = 0
                first_round_asp.append([])
            else:
                this_clean_answer = get_numeric_results(this_response)
                this_short_answer = []
                for path in this_clean_answer:
                    if (not path in this_short_answer):
                        this_short_answer.append(path)
                first_round_asp.append(this_short_answer)
                hit_count, fal_count = compare_results(this_short_answer, correct_paths)

                # if hit_count > len(correct_paths):
                    # print(e_idx)
                    # print(this_short_answer)
                    # print(correct_paths)
                    
            res_collections.append([hit_count, fal_count, len(correct_paths)])

        if problem_name == 'shortest_path':
            torch.save(first_round_asp, './all_shortest_path_first_round.pt')
    
        hit_ratios = []
        h_num = 0
        f_num = 0
        c_num = 0
        for elem in res_collections:
            hit_count, fal_count, correct_num = elem
            hit_ratio = hit_count / correct_num
            h_num += hit_count
            f_num += fal_count
            c_num += correct_num

            hit_ratios.append(hit_ratio)

        correct_ratio = h_num / c_num

        print(f'{f_num=} {h_num=}')
        print(f'comparison ratio (fal / hit) =  {f_num / h_num}')
        # print(max(hit_ratios))
        # print(sum(hit_ratios) / len(hit_ratios))
        # print(min(hit_ratios))
        print(f'correct ratio (h num / c num) = {correct_ratio}')

        if problem_name == 'shortest_path_verify':
            total_new_ans = 0
            for ee_idx in range(0, 50):
                prev_reponse = first_round_asp[ee_idx]
                curr_response = llm_refined_answer[ee_idx]
                curr_clean_answer = get_numeric_results(curr_response)
                curr_short_answer = []
                for path in curr_clean_answer:
                    if (not path in curr_short_answer):
                        curr_short_answer.append(path)
                
                for elem in curr_short_answer:
                    if (not elem in prev_reponse):
                        print(f'found new ans at {ee_idx}')
                        total_new_ans += 1

            print(f'number of new generated ans = {total_new_ans}')

    else:
        print('ERROR: PROBLEM NAME NOT DEFINED')
        return False

    return correct_ratio
import json
import numpy as np
DREAM_DIMS = ["ROT","Motivation", "Emotion", "Consequence"]
ATOMIC_DIMS = ["XAttribute","XEffect","XIntent","XNeed","XReact","XWant","OtherReact" ,
               "OtherWant","OtherAttribute","OtherEffect"]
atomic_dict = {}
dream_dict = {}
concept_dict  = {}
with open("samples\\annotated_data.jsonl") as f:
    total_dims = 0
    total_lines = 0
    total_atomic = 0
    total_dream = 0
    total_concept = 0
    dims_dict = {}
    for line in f.readlines():
        total_lines += 1
        d = json.loads(line)
        dims = d["chosen_dimensions"]
        l_dims = len(dims)
        total_dims += l_dims
        for di in dims:
            if di not in dims_dict:
                dims_dict[di] =0
            dims_dict[di] += 1
            if di in ATOMIC_DIMS:
                total_atomic += 1
                if di not in atomic_dict:
                    atomic_dict[di] = 0
                atomic_dict[di] +=1
            elif di in DREAM_DIMS:
                total_dream += 1
                if di not in dream_dict:
                    dream_dict[di] = 0
                dream_dict[di] +=1
            else:
                total_concept += 1
                if di not in concept_dict:
                    concept_dict[di] = 0
                concept_dict[di] +=1

    dims_list = [dims_dict[i] for i in dims_dict]
    print(f"The average number of dimensions per example is {total_dims / total_lines}")
    print(f"The average number of samples in which a dimension was used is {np.mean(dims_list)}")

    print(f"The average number of dimensions per example from atomic is {total_atomic / total_lines}")
    print(f"The average number of dimensions per example from dream is {total_dream / total_lines}")
    print(f"The average number of dimensions per example from conceptnet is {total_concept / total_lines}")

def print_dict(name, dict):
    print(f"for dict {name}")
    dims_list = [dims_dict[i] for i in dict]
    print(f"The average number of samples in which a dimension was used is {np.mean(dims_list)}")

print_dict("atomic", atomic_dict)
print_dict("dream", dream_dict)
print_dict("conceptnet", concept_dict)

import torch
from solvers.solver import Solver
from settings import INSTANCE_FOLDER
import numpy as np
from cpmp.layout import read_file, layout_to_tensors
import copy


class ModelSolver(Solver): 
    def __init__(self, model):
        super().__init__("ModelSolver")
        self.model = model
     
    def solve_from_path(self, instance_path, H, max_steps):
        layout = read_file(instance_path, H)
        start_unsorted_stacks = layout.unsorted_stacks
        
        # Conjunto para almacenar los estados visitados (como tuplas inmutables)
        visited_states = set()
        
        with torch.no_grad():
            while not layout.is_sorted():
                # Guardamos el estado actual antes de mover
                # Convertimos cada stack a tupla para que sea "hasheable"
                current_state = tuple(tuple(stack) for stack in layout.stacks)
                visited_states.add(current_state)

                G, P, I, S = layout_to_tensors(layout)
                GT = torch.from_numpy(G).unsqueeze(0)
                PT = torch.from_numpy(P).unsqueeze(0)
                IT = torch.from_numpy(I).unsqueeze(0)
                ST = torch.from_numpy(np.array([S])).unsqueeze(0)
                HT = torch.from_numpy(np.array([H])).unsqueeze(0)    

                logits = self.model(GT, PT, IT, ST, HT)
                
                # Ordenamos todos los índices de mejor a peor
                _, top_indices = torch.sort(logits, dim=1, descending=True)
                top_indices = top_indices.squeeze(0)

                for i in range(len(top_indices)):
                    best_index = top_indices[i].item()
                    src = int(best_index / (S-1))
                    r = best_index % (S-1)
                    dst = r if r < src else r + 1

                    # 1. Previsualizamos el movimiento con deepcopy
                    temp_layout = copy.deepcopy(layout)
                    temp_layout.move(src, dst)
                    next_state = tuple(tuple(stack) for stack in temp_layout.stacks)
                    
                    # 2. Verificamos si el estado resultante ya fue visitado
                    if next_state not in visited_states:
                        layout.move(src, dst)
                        break

                if layout.steps >= max_steps:
                    break

        solved = layout.unsorted_stacks == 0
        return solved, layout.steps
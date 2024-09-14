import random
import torch
import matplotlib.pyplot as plt

vowel_list = set(["A", "E", "I", "İ", "O", "Ö", "U", "Ü"])

def read_txt(path:str)-> list:
    names = open(path, "r").read().splitlines()
    return names


def random_name_sampling(names:list, number=5):
    print("\n----------------------------\n")
    print("Random choosen names:  \n")
    for i in range(number):
        print(random.choice(names))
    print("\n----------------------------\n")

def get_max_min_name_count(names:list):
    min_counter = min(len(w) for w in names)
    max_counter = max(len(w) for w in names) 
    print("\n----------------------------\n")
    print(f"Minimum name count: {min_counter}")
    print(f"Maximum name count: {max_counter}")
    print("\n----------------------------\n")
    return min_counter, max_counter
def preprocessing_names(names:list)->list:
    new_names = [name for name in names if " " not in name]
    return new_names

def tokenize_letters(names:list):
    chars = sorted(list(set(''.join(names))))
    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    return itos, stoi

def create_combination_matrix(itos:dict, stoi:dict, names:list)->torch.Tensor:
    N = torch.zeros((len(itos),len(itos)), dtype = torch.int32)
    for w in names:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] +=1
    return N

def visualize_matrix(N:torch.Tensor, itos:dict):
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')

    for i in range(len(itos)):
        for j in range(len(itos)):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
            
    plt.axis('off')
    plt.show()

def generate_names(N:torch.Tensor, itos:dict, max_name:int, min_name:int):
    print("Generated names: ")
    P = (N+1).float()
    P = P/ P.sum(1, keepdim = True)
    for i in range(50):
        ix = 0
        out = []
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True).item()
            out.append(itos[ix])
            if ix == 0:
                break
        if len(out) < max_name and len(out) > min_name+1:
            char_list = out[:-1]
            is_valid = 1  # Başlangıçta is_valid'i 1 olarak ayarlıyoruz
            
            # Ardışık üç sessiz harf kontrolü
            for j in range(len(char_list) - 2):
                three_chars = char_list[j:j+3]
                
                # Eğer üç harf de sessizse, is_valid'i 0 yap ve kontrolü kır
                if all(char not in vowel_list for char in three_chars):
                    is_valid = 0
                    break
                
                # En az bir sesli harf varsa is_valid 1 kalır
                if any(char in vowel_list for char in three_chars):
                    is_valid = 1

            # Eğer valid bir isimse yazdır
            if is_valid:
                print(''.join(char_list))

if __name__ == "__main__":
    names = read_txt("isimler2")
    names = preprocessing_names(names)
    random_name_sampling(names, 10)
    min_counter, max_counter = get_max_min_name_count(names)
    itos, stoi = tokenize_letters(names)
    N = create_combination_matrix(itos, stoi, names)
    #visualize_matrix(N, itos)
    generate_names(N, itos, max_counter, min_counter)
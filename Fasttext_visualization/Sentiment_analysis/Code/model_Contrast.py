import numpy as np

wa_nft = np.loadtxt("WA_NFT.txt")
wa_ft = np.loadtxt("WA_FT.txt")
wa_cnn = np.loadtxt("WA_CNN.txt")
# wa_nft3 = np.loadtxt("WA_NFT3.txt")
wa_gru = np.loadtxt("WA_GRU.txt")

num = 0
wa_nft_class = [i[0] for i in wa_nft]
wa_cnn_class = [i[0] for i in wa_cnn]
print(wa_nft_class)
print(wa_cnn_class)
for i in wa_nft_class:
    if i not in wa_cnn_class:
        num = num + 1
ans = num / len(wa_nft)
print(num, len(wa_nft), ans)

num = 0
wa_nft_class = [i[0] for i in wa_nft]
wa_gru_class = [i[0] for i in wa_gru]

for i in wa_nft_class:
    if i not in wa_gru_class:
        num = num + 1
ans = num / len(wa_nft)
print(num, len(wa_nft), ans)








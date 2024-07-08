import json
import numpy as np
import matplotlib.pyplot as plt


cond_for_50_ts = json.load(open("/home/test/rishubh/sachi/CelebBasis_pstar/cond_list_for_each_timesteps.json",))
cond_for_ts = json.load(open("/home/test/rishubh/sachi/CelebBasisv1/cond_for_each_timestep.json",))

cond_for_50_ts = np.array(cond_for_50_ts['cond_list'])
cond_for_ts = np.array(cond_for_ts["cond_list"])

print(cond_for_50_ts.shape) # [50,2,768]
print(cond_for_ts.shape) # [1,2,768]

es = 2 

cosine_values = []
for i in range(es):
    for j in range(50):
        cosine_values.append(np.dot(cond_for_50_ts[j,i,:],cond_for_ts[0,i,:])/(np.linalg.norm(cond_for_50_ts[j,i,:])*np.linalg.norm(cond_for_ts[0,i,:])))
print(len(cosine_values))

# plot the cosine similarity values as heatmap
cosine_values = np.array(cosine_values)
cosine_values = cosine_values.reshape(es,50)

plt.imshow(cosine_values)
plt.colorbar()
plt.savefig("cosine_similarity_between_timesteps.png")
   
cosine_values = []
for i in range(es):
    for j in range(50):
        for k in range(50):
            cosine_values.append(np.dot(cond_for_50_ts[j,i,:],cond_for_50_ts[k,i,:])/(np.linalg.norm(cond_for_50_ts[j,i,:])*np.linalg.norm(cond_for_50_ts[k,i,:])))
print(len(cosine_values))
cosine_sim1 = np.array(cosine_values[:2500])
cosine_sim2 = np.array(cosine_values[2500:])
cosine_sim1 = cosine_sim1.reshape(50,50)
cosine_sim2 = cosine_sim2.reshape(50,50)

plt.clf()
plt.imshow(cosine_sim1)
plt.colorbar()
plt.savefig("cosine_similarity_between_timesteps_1.png")

plt.clf()
plt.imshow(cosine_sim2)
plt.colorbar()
plt.savefig("cosine_similarity_between_timesteps_2.png")

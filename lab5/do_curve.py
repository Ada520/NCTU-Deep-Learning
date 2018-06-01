import matplotlib.pyplot as plt
import numpy as np

mean_reward=[]
max_reward=[]
fp = open('mean_td0.txt', "r")
line = fp.readline()

while line:
	mean_reward.append(float(line))
	line = fp.readline()
fp.close()

fp = open('max_td0.txt', "r")
line = fp.readline()

while line:
	max_reward.append(float(line))
	line = fp.readline()
fp.close()

print(max(mean_reward))
print(max(max_reward))

plt.title('TD0 reward')
plt.xlabel('episode(1000)')
plt.ylabel('reward')
plt.plot(np.arange(1,len(mean_reward)+1),mean_reward , color = 'blue',label='mean_reward')
plt.plot(np.arange(1,len(max_reward)+1),max_reward , color = 'red',label='max_reward')
plt.legend(loc='upper right')
savefilename = 'TD0-reward.jpg'
plt.savefig(savefilename)
plt.show()
plt.close()
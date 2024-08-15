import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


if __name__=="__main__":
    data_path = r'./Data/'
    result_path = r'./Results/Semantic'
    sub_ids = ['sub-%02d'%i for i in range(1,16)]

    nfolds = 10 #number of folds for cross-validation
    nrands = 1000 #number of randomization interations

    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression(n_jobs=5)
    
    #Load word embeddings
    with open(f'{data_path}/Semantic/words.txt','r') as f:
        words = f.readlines()
    words = np.array([w.strip() for w in words])
    vecs = np.load(f'{data_path}/Semantic/vecs.npy')

    for pti, pt in enumerate(sub_ids):

        print(f'Running participant: {pt}')

        #Load the data
        data = np.load(f'{data_path}/Semantic/{pt}_features.npy')
        labels = np.load(f'{data_path}/Semantic/{pt}_labels.npy')
        
        #Load the channel names
        channels = np.load(f'{data_path}/General/{pt}_channel_names.npy')

        #Get individual words
        markers = np.concatenate([[-1],np.where(np.diff(labels[:,None]==b'',axis=0)!=0)[0]])
        trialData = []
        trialLabel = []
        trialEmbeddings = []
        for i in range(0,len(markers)-1,2):
            lbl = labels[markers[i]+1].astype(str).strip('\r')
            if lbl in words:
                trialEmbeddings.append(vecs[np.argwhere(words==lbl)[0][0],:])
                trialData.append(data[markers[i]+1:markers[i+1],:])
                trialLabel.append(lbl)
        trialEmbeddings = np.array(trialEmbeddings)

        #Downsample
        shortestTrial = np.min([a.shape[0] for a in trialData])
        downFactor = 1
        shortestTrial = shortestTrial - (shortestTrial%downFactor)
        trialData = np.array([a[:shortestTrial,:].reshape(int(shortestTrial/downFactor),downFactor,data.shape[-1]).mean(axis=1) for a in trialData])

        #Initialize an empty array to save the results to
        rec_trial = np.zeros(trialData.shape)
        rng = np.random.default_rng()
        coefs = np.zeros((nfolds,trialData.shape[1],trialEmbeddings.shape[1],trialData.shape[2]))
        
        ### Correlations ###
        for k,(train, test) in enumerate(kf.split(trialEmbeddings)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(trialData[train,:],axis=(0,1))
            std=np.std(trialData[train,:],axis=(0,1))
            trainData=(trialData[train,:]-mu)/std
            testData=(trialData[test,:]-mu)/std
            
            #Proceed for each channel individually
            for ch in range(trainData.shape[-1]):
                #Fit the regression model
                est.fit(trialEmbeddings[train,:], trainData[:,:,ch])
                coefs[k,:,:,ch]=est.coef_
                #Predict the neural data
                rec_trial[test, :,ch] = est.predict(trialEmbeddings[test,:])

        #Calculate correlations
        rs=np.zeros(trialData[0,:,:].shape)
        for ch in range(trialData.shape[-1]):
            for ts in range(trialData.shape[1]):
                rs[ts,ch],_=pearsonr(trialData[:,ts,ch],rec_trial[:,ts,ch])

        ### Permutations ###
        #Estimate random baseline
        randomRs = np.zeros((nrands,trialData.shape[1],trialData.shape[-1]))
        mu=np.mean(trialData[train,:],axis=(0,1))
        std=np.std(trialData[train,:],axis=(0,1))
        trainData=(trialData-mu)/std
        dflat = trainData.reshape(trainData.shape[0]*trainData.shape[1],trainData.shape[2])
        for ch in range(trainData.shape[-1]):
            est.fit(trialEmbeddings, trainData[:,:,ch])
            for randRound in range(nrands):
                randEmbeddings = rng.permutation(trialEmbeddings,axis=1)
                rec_trial= est.predict(randEmbeddings)
                rec = rec_trial.flatten()
                for ts in range(trainData.shape[1]):
                    randomRs[randRound,ts,ch],_ = pearsonr(trainData[:,ts,ch],rec_trial[:,ts])

        ### Significance ###
        alpha= 0.05
        threshold = np.sort(np.max(randomRs,axis=(1,2)))[-int(randomRs.shape[0]*alpha)]
        sigs = rs > threshold
        
        #Preview results
        print(f'{pt} | {np.sum(sigs)} significant points in {np.sum(np.max(sigs,axis=0))} channels with highest r of {np.max(rs)} [in a total of {trialData.shape[0]} words]')

        #Save the results  
        os.makedirs(result_path,exist_ok=True)    
        np.save(os.path.join(result_path,f'{pt}_correlations.npy'), rs)
        np.save(os.path.join(result_path,f'{pt}_permutations.npy'), randomRs)
        np.save(os.path.join(result_path,f'{pt}_sigs.npy'), sigs)

        print('done')
print('done')
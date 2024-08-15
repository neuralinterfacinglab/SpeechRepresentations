import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from pandas import read_csv


if __name__=="__main__":
    data_path = r'./Data'
    result_path = r'./Results/Articulatory'
    sub_ids = ['sub-%02d'%i for i in range(1,16)]

    nfolds = 10 #number of folds for cross-validation
    nframes = 9 #temporal context (timeframes)
    nrands = 1000 #number of randomization interations

    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression()

    for pti, pt in enumerate(sub_ids):
        
        print(f'Running participant: {pt}')
        
        #Load articulatory data
        all_trajectories = read_csv(f'{data_path}/Articulatory/{pt}_trajectories.csv')

        #Load the eeg features
        eeg = np.load(f'{data_path}/Articulatory/{pt}_features.npy')

        #Load the channel names
        channels = np.load(f'{data_path}/General/{pt}_channel_names.npy')

        #Remove subglottal pressure
        trajectories = np.array(all_trajectories)[:,1:]
        narticulators = trajectories.shape[1]

        #Initialize result arrays
        predictions = np.zeros((nframes,len(channels),eeg.shape[0],narticulators),dtype='uint8')
        correlations = np.zeros((nframes,len(channels),nfolds,narticulators))
        
        ### Correlations ###
        for tf in range(nframes):

            #Extract data from 1 timeframe
            cuton = int(eeg.shape[1]/9*(tf))
            cutoff = int(eeg.shape[1]/9*(tf+1))
            data = eeg[:,cuton:cutoff]            
            
            #Do a cross-validation across all channels
            for ch, channel in enumerate(channels):
            
                #Select the features of this channel
                feats = data[:,ch]
                
                for k,(train, test) in enumerate(kf.split(feats)):
                    #Z-Normalize with mean and std from the training data
                    mu=np.mean(feats[train],axis=0)
                    std=np.std(feats[train],axis=0)
                    trainData=(feats[train]-mu)/std
                    testData=(feats[test]-mu)/std

                    #Fit the regression model
                    est.fit(trainData.reshape(-1,1), trajectories[train,:])
                    #Predict the reconstructed trajectories for the test data
                    prediction = est.predict(testData.reshape(-1,1))
                    #Save the fold
                    predictions[tf,ch,test,:] = prediction
                    #Calculate correlations
                    for i in range(prediction.shape[1]):
                        r, p = pearsonr(trajectories[test,i], prediction[:,i])
                        correlations[tf,ch,k,i] = r

            #Preview results        
            print(f'{pt} | timeframe {tf+1} has maximum correlation of {np.max(np.mean(correlations[tf,:,:,:], axis=2))}')

        ### Permutations ###
        #Only include subglottal pressure
        trajectory = np.array(all_trajectories)[:,0]

        #Initialize permutation results array
        permutations = np.zeros((len(channels), nrands, nfolds))

        #Extract data from 1 timeframe
        cuton = int(eeg.shape[1]/9*(4))
        cutoff = int(eeg.shape[1]/9*(5))
        data = eeg[:,cuton:cutoff]            
        
        for randRound in range(nrands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(data.shape[0]*0.1),int(data.shape[0]*0.9)))
            #Swap the eeg on the splitting point 
            shuffled = np.concatenate((data[splitPoint:,:],data[:splitPoint,:]))  
            #Do a cross-validation
            for k,(train, test) in enumerate(kf.split(shuffled)):
                #Calculate correlations
                corr_matrix = np.corrcoef(shuffled[test,:].T, trajectory[test].T)
                fold = np.mean(corr_matrix[:shuffled.shape[1], shuffled.shape[1]:], axis=1)
                permutations[:,randRound,k] = fold

        #Average across folds
        final = np.mean(permutations, axis=2) 

        ### Significance ###   
        threshold = np.max([np.percentile(final[ch,:],95) for ch in range(final.shape[0])])         
        total_sigs = np.ones((nframes, len(channels), correlations.shape[3]), dtype='bool')
        sigs = np.ones((nframes, len(channels)), dtype='bool')
        for tf in range(nframes):
            for a in range(correlations.shape[3]):
                for ch in range(len(channels)):
                    corr = np.mean(correlations[tf,ch,:,a])
                    if corr > threshold:
                        total_sigs[tf,ch,a] = True
                    else:
                        total_sigs[tf,ch,a] = False     
            #Combine articulators            
            sigs[tf,:] = [True if True in total_sigs[tf,ch,:] else False for ch in range(total_sigs.shape[1])]
        
        #Save the results
        os.makedirs(result_path,exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predictions.npy'), predictions) 
        np.save(os.path.join(result_path,f'{pt}_correlations.npy'), correlations)
        np.save(os.path.join(result_path,f'{pt}_articulator_names.npy'), np.array(all_trajectories.columns[1:]))
        np.save(os.path.join(result_path,f'{pt}_permutations.npy'), final)
        np.save(os.path.join(result_path,f'{pt}_sigs.npy'), sigs)  

        print('done')
print('done')
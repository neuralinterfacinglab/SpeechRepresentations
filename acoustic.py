import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


if __name__=="__main__":
    data_path = r'./Data/'
    result_path = r'./Results/Acoustic'
    sub_ids = ['sub-%02d'%i for i in range(1,16)]

    nfolds = 10 #number of folds for cross-validation
    nframes = 9 #temporal context (timeframes)
    nrands = 1000 #number of randomization interations
    
    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression()

    for pti, pt in enumerate(sub_ids):
        
        print(f'Running participant: {pt}')

        #Load the spectrogram
        spectrogram = np.load(f'{data_path}/Acoustic/{pt}_spectrogram.npy')
        nbins = spectrogram.shape[1]

        #Load the eeg features
        eeg = np.load(f'{data_path}/Acoustic/{pt}_features.npy')

        #Load the channel names
        channels = np.load(f'{data_path}/General/{pt}_channel_names.npy')

        #Initialize result arrays
        predictions = np.zeros((nframes,len(channels),eeg.shape[0],nbins),dtype='uint8')
        correlations = np.zeros((nframes,len(channels),nfolds,nbins))
        
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
                    est.fit(trainData.reshape(-1,1), spectrogram[train,:])
                    #Predict the reconstructed trajectories for the test data
                    prediction = est.predict(testData.reshape(-1,1))
                    #Save the fold
                    predictions[tf,ch,test,:] = prediction
                    #Calculate correlations
                    for i in range(prediction.shape[1]):
                        r, p = pearsonr(spectrogram[test,i], prediction[:,i])
                        correlations[tf,ch,k,i] = r

            #Preview of results        
            print(f'{pt} | timeframe {tf+1} has maximum correlation of {np.max(np.mean(correlations[tf,ch,:,:]))}')

        ### Permutations ###
        #Initialize permutation results array
        permutations = np.zeros((len(channels), nrands, nfolds))
        
        #Extract data from 1 timeframe
        cuton = int(eeg.shape[1]/9*(5-1))
        cutoff = int(eeg.shape[1]/9*5)
        data = eeg[:,cuton:cutoff]   
        
        #Run permutations         
        for randRound in range(nrands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(data.shape[0]*0.1),int(data.shape[0]*0.9)))
            #Swap the eeg on the splitting point 
            shuffled = np.concatenate((data[splitPoint:,:],data[:splitPoint,:]))  
            #Do a cross-validation
            for k,(train, test) in enumerate(kf.split(shuffled)):
                #Calculate correlations
                corr_matrix = np.corrcoef(shuffled[test,:].T, spectrogram[test,:].T)
                fold = np.mean(corr_matrix[:shuffled.shape[1], shuffled.shape[1]:], axis=1)
                permutations[:,randRound,k] = fold
        
        #Average across folds
        final = np.mean(permutations, axis=2)

        ### Significance ###
        threshold = np.max([np.percentile(final[ch,:],95) for ch in range(final.shape[0])])
        sigs = np.ones((nframes, len(channels)), dtype='bool')
        for tf in range(nframes):
            results = [np.mean(correlations[tf,ch,:,:]) for ch in range(len(channels))]
            sigs[tf,:] = [True if corr>threshold else False for corr in results]
        
        #Save the results
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predictions.npy'), predictions) 
        np.save(os.path.join(result_path,f'{pt}_correlations.npy'), correlations)
        np.save(os.path.join(result_path,f'{pt}_permutations.npy'), final)
        np.save(os.path.join(result_path,f'{pt}_sigs.npy'), sigs)  

        print('done')
print('done')
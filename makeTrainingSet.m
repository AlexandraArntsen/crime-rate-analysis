function [Ytrain Xtrain Ytest Xtest] = makeTrainingSet(Target,Predictors,split) 
% split is size of test set value is 0 to 1

rng(5); % For reproducibility

  [m n] = size(Predictors); 
   part = cvpartition(m,'holdout',split);
istrain = training(part); % data for fitting
 istest = test(part); % data for quality assessment

train_ind = find(istrain == true);
    Ytrain = Target(train_ind,:); 
    Xtrain = Predictors(train_ind,:); 
 test_ind = find(istest == true);
    Ytest = Target(test_ind,:); 
    Xtest = Predictors(test_ind,:); 
end



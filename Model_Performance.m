
function [R2_test, R2_train] = Model_Performance(mdl,ypred_train,Ytrain,ypred_test,Ytest,Xtrain,Xtest)
%------------------------------------------------%
%                                                %
%          Eval model performance                %
%     .Candidate Exam for Data Scientist.        %
%               .NRG Systems.                    %
%                                                %
%------------------------------------------------%

    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'LinearLeastSquares' );
    
    % check performance on train data
    [fitresult, gof] = fit(Ytrain,ypred_train, ft,opts); 
    R2_train = gof.rsquare;
    fprintf('R2 training: %f\n\n',R2_train)
    
    % check performance on test data       
    [fitresult, gof] = fit(Ytest,ypred_test, ft,opts);
    R2_test = gof.rsquare;
    fprintf('R2 test: %f\n\n',R2_test)

 if isempty(mdl) == false   
   figure
   subplot(1,2,1)
   plot(Ytrain,'b','LineWidth',2), hold on
   plot(predict(mdl,Xtrain(:,:)),'r.-','LineWidth',1,'MarkerSize',15)
   title('performance on training data')
   legend({'Actual','Predicted'})
   xlabel('Training Data point');
   ylabel('Crime Rate');
   set(gca, 'FontSize', 16) 
   subplot(1,2,2)
   scatter(Ytrain,predict(mdl,Xtrain(:,:)))
   hold on
   plot([0,20],[0,20], '-k')
   title('performance on training data')
   legend({'sample'})
   xlabel('true crime rate');
   ylabel('predicted crime Rate');
   set(gca, 'FontSize', 16) 

  figure
  subplot(1,2,1)
  plot(Ytest,'b','LineWidth',2), hold on
  plot(predict(mdl,Xtest(:,:)),'r.-','LineWidth',1,'MarkerSize',15)
  title('performance on test data')
  legend({'Actual','Predicted'})
  xlabel('Test Data point');  
  ylabel('Crime Rate');
  set(gca, 'FontSize', 16) 
  subplot(1,2,2)
  scatter(Ytest,predict(mdl,Xtest(:,:)))
  hold on
  plot([0,20],[0,20], '-k')
  title('performance on test data')
  legend({'sample'})
  xlabel('true crime rate');
  ylabel('predicted crime Rate');
  set(gca, 'FontSize', 16) % Observe first hundred points, pan to view more

elseif isempty(mdl) == true  
   figure
   subplot(1,2,1)
   plot(Ytrain,'b','LineWidth',2), hold on
   plot(ypred_train,'r.-','LineWidth',1,'MarkerSize',15)
   title('performance on training data')
   legend({'Actual','Predicted'})
   xlabel('Training Data point');
   ylabel('Crime Rate');
   set(gca, 'FontSize', 16) 
   subplot(1,2,2)
   scatter(Ytrain,ypred_train)
   hold on
   plot([0,20],[0,20], '-k')
   title('performance on training data')
   legend({'sample'})
   xlabel('true crime rate');
   ylabel('predicted crime Rate');
   set(gca, 'FontSize', 16) 

  figure
  subplot(1,2,1)
  plot(Ytest,'b','LineWidth',2), hold on
  plot(ypred_test,'r.-','LineWidth',1,'MarkerSize',15)
  title('performance on test data')
  legend({'Actual','Predicted'})
  xlabel('Test Data point');  
  ylabel('Crime Rate');
  set(gca, 'FontSize', 16) 
  subplot(1,2,2)
  scatter(Ytest,ypred_test)
  hold on
  plot([0,20],[0,20], '-k')
  title('performance on test data')
  legend({'sample'})
  xlabel('true crime rate');
  ylabel('predicted crime Rate');
  set(gca, 'FontSize', 16) % Observe first hundred points, pan to view more
 end
 

  
end

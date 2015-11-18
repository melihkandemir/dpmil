function [ypred,probs]=dpmil_instance_predict(Xts,model)
  
    addpath ../../libsvm/
   
    p0=predict_gmm(Xts,model.model0);
    p1=predict_gmm(Xts,model.model1);     
    probs=p1./(p1+p0);
    ypred=-1+2*(probs>0.5);
   
end
     

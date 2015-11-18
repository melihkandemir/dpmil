function [bpred,bagprob]=dpmil_predict(Xts,bags,model)
    addpath ../libsvm/
    
   baglist=unique(bags);
 
   for bb = 1:length(baglist)
       Xbag = Xts(bags==baglist(bb),:);
       [Nb,D]=size(Xbag);
      
       p0=predict_gmm(Xbag,model.model0)*(Nb-1/Nb);
       p1=predict_gmm(Xbag,model.model1)*(1/Nb);     
       probs=p1./(p1+p0);
       bagprob(bb)=max(probs);          
        
       bpred(bb)=-1+2*max(probs>0.5);
        
   end
    
    bpred=bpred';
    bagprob=bagprob';
  
end
     

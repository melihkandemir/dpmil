function pk=predict_gmm(Xnew,model)
    
    [N,D]=size(Xnew);
   
    
    pk=zeros(N,model.K);
    for kk=1:model.K
     %    pk(:,kk)=model.alpha(kk)*mvnpdf(Xnew,model.m(:,kk)',eye(D)*0.5)/sum(model.alpha);
        pk(:,kk)=model.alpha(kk)*mvnpdf(Xnew,model.m(:,kk)',model.M(:,:,kk))/sum(model.alpha);
     
    end
    pk=sum(pk,2);
end
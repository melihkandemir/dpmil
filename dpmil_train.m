function finalmodel=dpmil_train(X,y,bags,opt)
 
        ytrue=y;

        %[X,indices,y,yinst,ybag]=struct_to_concat(traindata);


        baglist=unique(bags);
        Nbags=length(baglist);
       
        y=[];

        for bb=1:Nbags
            y=[y; ytrue(bb)*ones(sum(bags==baglist(bb)),1);];

        end
        indices=bags;
    
        r=1*(y>0);
         
        % bagcounts=find_bag_sizes(X,indices);
         positiveClassPrior=0.5; %1./(bagcounts);
        
        [N,D]=size(X);
        
        if sum(r==0)<opt.K
            opt.K=floor(sum(r==0)*0.5);
        end
        
        if sum(r==1)<opt.K
            opt.K=floor(sum(r==1)*0.5);
        end        
        
        model0=initmodel(X(r==0,:),opt.K,opt);
        model1=initmodel(X(r==1,:),opt.K,opt);
        
        prevM0=zeros(D,model0.K);
        prevM1=zeros(D,model1.K);
        
        prevr=zeros(N,1);
        
        cnt=0;
        L=zeros(2,1);

        for ii=1:opt.maxiter
            fprintf('.')
            if mod(ii,10)==0
                fprintf(' %d\n',uint16(ii));
            end
   
           model0=vexp(X, model0);
           model1=vexp(X, model1);   
                
            p0=predict_gmm(X,model0).*(1-positiveClassPrior);
            p1=predict_gmm(X,model1).*positiveClassPrior;


            rprev=r;
            r=p1./(p0+p1);
            r(isnan(r))=0;
            r(ytrue==0)=0;
            
            for bb=1:Nbags
                if max(r(indices==baglist(bb))>0.5)==0 && ytrue(bb)==1
                    rprev_bb=rprev(indices==baglist(bb));
                    ybb=r(indices==baglist(bb));
                    Nb=length(ybb);                    
                    posinst_prev=find(rprev_bb>0.5);
                    rbb=zeros(Nb,1);
                    
                    [~,idx]=sort(ybb(posinst_prev),'descend');
                    rbb(posinst_prev(idx(1)))=1;
                    
                    r(indices==baglist(bb))=rbb;
                end
            end                                     
            r(ytrue==0)=0;
            model1.r=r;
            model0.r=r;

	       
            
            model0.logR=model0.logR(r<=0.5,:);
            model0.R=model0.R(r<=0.5,:);                  
            
            model1.logR=model1.logR(r>0.5,:);
            model1.R=model1.R(r>0.5,:);     
                        
            model0=vmax(X(r<=0.5,:), model0, model0.prior);
            model1=vmax(X(r>0.5,:), model1, model1.prior);
            
            %if mod(ii,5)==0
                cnt=cnt+1;
                L(cnt) = log(-vbound(X(r<=0.5,:)', model0, model0.prior)-vbound(X(r>0.5,:)', model1, model1.prior));
            %end
           
                                
            if sum(sum(abs(model1.m-prevM1)))+sum(sum(abs(model0.m-prevM0)))<0.001 && sum(abs(prevr-r))<0.001;
                break;
            else
                prevM1=model1.m;
                prevM0=model0.m;
                prevr=r;
            end
            
           % fprintf('Perc: %.5f\n',mean(r(ytrue==1)>0.5));
        end
        
        model1.r=r;
        
        y(model1.r>0.5)=1;
        y(model1.r<=0.5)=-1;

        
        fprintf('\n Label inference is complete with %.5f positive labels.\n Now training the classifier.',mean(r(ytrue==1)));
        
        finalmodel.model1=model1;
        finalmodel.model0=model0;
        finalmodel.L=L;
end

function model=initmodel(X,K,opt)
    [N,D]=size(X);
    model.K=K;

    model.R=ones(N,K)/K;

    model.v=ones(1,K)*D;
    model.alpha=ones(1,K)/K;
    model.kappa=ones(1,K);
    model.beta=ones(2,K); % stick-breaking VB parameters    
        
   Kern=X*X'; % start from the data points that are most different from others
   [~,simorder]=sort(sum(Kern,2));   
  
     model.m=X(simorder(1:K),:)';
  
    
    for kk=1:K
        model.M(:,:,kk)=eye(D)/K;
    end
    
    model.prior.kappa = 1;
    model.prior.m = mean(X)';
    model.prior.v = D+1;
    model.prior.M = cov(X)+0.001*eye(D);  
    model.prior.alpha1=1;
    model.prior.alpha2=opt.alpha;
    model.prior.alpha=1;
    
end

function model = vmax(X, model, prior)
    X=X';
    
    
    kappa0 = prior.kappa;
    m0 = prior.m;
    v0 = prior.v;
    M0 = prior.M;
    R = model.R;

    nk = sum(R,1); % 10.51
    
    
    nxbar = X*R;
    kappa = kappa0+nk; % 10.60
    kappa2=kappa;
    kappa2(kappa<0.0001)=0.0001;
    m = bsxfun(@times,bsxfun(@plus,kappa0*m0,nxbar),1./kappa2); % 10.61
    v = v0+nk; % 10.63

    [d,k] = size(m);
    M = zeros(d,d,k); 
    sqrtR = sqrt(R);

    nk2=nk;
    nk2(nk<0.0001)=0.0001;
    xbar = bsxfun(@times,nxbar,1./nk2); % 10.52
    xbarm0 = bsxfun(@minus,xbar,m0);
    w = (kappa0*nk./(kappa0+nk2));
    
    beta_vb=zeros(2,k);
    alpha=ones(1,k);
    
    for i = 1:k
        Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
        xbarm0i = xbarm0(:,i);
        
       
        Mi = (M0+Xs*Xs'+w(i)*(xbarm0i*xbarm0i')); % 10.62
        Mi = safecov(Mi);
        %Mi(eye(d)==1)=diag(Mi);
        
        beta_vb(1,i) = model.prior.alpha1+sum(R(:,i));
        beta_vb(2,i) = model.prior.alpha2+sum(R((i+1):end,i));        
        
        M(:,:,i)=Mi;
             
    end    
 

    model.beta = beta_vb;
    
    E1minV=model.beta(2,:)./sum(model.beta);
    for i = 1:k
        if i>1
            alpha(i)=prod(E1minV(1:(i-1)));
        end
        alpha(i)=alpha(i)*model.beta(1,i)/sum(model.beta(:,i));    
    end
    
    model.alpha = alpha;
    model.kappa = kappa;
    model.m = m;
    model.v = v;
    model.M = M;
end

function bagcnt=find_bag_sizes(X,indices)
  bagcnt=zeros(size(X,1),1);
  indList=unique(indices);
  
  for ii=1:length(indList)
      cnt=sum(indices==indList(ii));
      bagcnt(indices==indList(ii))=cnt;
  end
end

function model = vexp(X, model)
    X=X';
    kappa = model.kappa;
    m = model.m; 
    v = model.v;
    M = model.M;
    beta_vb = model.beta;
    

    n = size(X,2);
    [d,k] = size(m);

    logW = zeros(1,k);
    EQ = zeros(n,k);
    Elogpi=zeros(1,k);
    Elog1minpi=zeros(1,k);
    for i = 1:k
        U = chol(M(:,:,i));
        logW(i) = -2*sum(log(diag(U)));      
        Q = (U'\bsxfun(@minus,X,m(:,i)));
        EQ(:,i) = d/kappa(i)+v(i)*dot(Q,Q,1);    % 10.64
        Elog1minpi(i)=psi(0,beta_vb(2,i))-psi(0,sum(beta_vb(:,i)));
        if i>1
            Elogpi(i)=sum(Elog1minpi(1:(i-1)));
        end
        Elogpi(i)=Elogpi(i)+psi(0,beta_vb(1,i))-psi(0,sum(beta_vb(:,i)));
    end

    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65

    logRho = (bsxfun(@minus,EQ,2*Elogpi+ElogLambda-d*log(2*pi)))/(-2); % 10.46
    logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
    R = exp(logR);
    
    
    logA = (bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)))/(-2) .* R; % update r
    

    model.logR = logR;
    model.R = R;
    model.logA=sum(logA,2);
end

function L = vbound(X, model, prior)
    alpha0 = prior.alpha;
    kappa0 = prior.kappa;
    m0 = prior.m;
    v0 = prior.v;
    M0 = prior.M;

    alpha = model.alpha; % Dirichlet
    kappa = model.kappa;   % Gaussian
    m = model.m;         % Gasusian
    v = model.v;         % Whishart
    M = model.M;         % Whishart: inv(W) = V'*V
    R = model.R;
    logR = model.logR;


    [d,k] = size(m);
    nk = sum(R,1); % 10.51

    Elogpi = psi(0,alpha)-psi(0,sum(alpha));

    Epz = dot(nk,Elogpi);
    Eqz = dot(R(:),logR(:));
    logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
    Eppi = logCalpha0+(alpha0-1)*sum(Elogpi);
    logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
    Eqpi = dot(alpha-1,Elogpi)+logCalpha;
    L = Epz-Eqz+Eppi-Eqpi;


    U0 = chol(M0);
    sqrtR = sqrt(R);
    xbar = bsxfun(@times,X*R,1./nk); % 10.52

    logW = zeros(1,k);
    trSW = zeros(1,k);
    trM0W = zeros(1,k);
    xbarmWxbarm = zeros(1,k);
    mm0Wmm0 = zeros(1,k);
    for i = 1:k
        U = chol(M(:,:,i));
        logW(i) = -2*sum(log(diag(U)));      

        Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
        V = chol(Xs*Xs'/nk(i)+0.001*eye(d));
        Q = V/U;
        trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
        Q = U0/U;
        trM0W(i) = dot(Q(:),Q(:));

        q = U'\(xbar(:,i)-m(:,i));
        xbarmWxbarm(i) = dot(q,q);
        q = U'\(m(:,i)-m0);
        mm0Wmm0(i) = dot(q,q);
    end

    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
    Epmu = sum(d*log(kappa0/(2*pi))+ElogLambda-d*kappa0./kappa-kappa0*(v.*mm0Wmm0))/2;
    logB0 = v0*sum(log(diag(U0)))-0.5*v0*d*log(2)-logmvgamma(0.5*v0,d);
    EpLambda = k*logB0+0.5*(v0-d-1)*sum(ElogLambda)-0.5*dot(v,trM0W);

    Eqmu = 0.5*sum(ElogLambda+d*log(kappa/(2*pi)))-0.5*d*k;
    logB =  -v.*(logW+d*log(2))/2-logmvgamma(0.5*v,d);
    EqLambda = 0.5*sum((v-d-1).*ElogLambda-v*d)+sum(logB);

    EpX = 0.5*dot(nk,ElogLambda-d./kappa-v.*trSW-v.*xbarmWxbarm-d*log(2*pi));

    L = L+Epmu-Eqmu+EpLambda-EqLambda+EpX;
end


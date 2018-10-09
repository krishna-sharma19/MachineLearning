[w,aps] = hard_neg();

function [w,aps]=hard_neg()
    
    [trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
    
    X = trD;
    y = trLb;

    [w,b,alpha,objVal]= svm_qp(X, y, valD, valLb,0.01);

    HW2_Utils.genRsltFile(w,b,"val","out")
    ap = HW2_Utils.cmpAP("out","val")

    epochs = 10;
    aps = zeros(epochs,1);
    objs = zeros(epochs,1)
    st = tic;
    
    for i=1:epochs        
        hog_vectors = mine(w,b,'trainIms');
        [~,N_1] = size(hog_vectors);
        X = cat(2,X,hog_vectors);
        y = cat(1,y,-1*ones(N_1,1));
        
        [w,b,alpha,objVal]= svm_qp(X, y, valD, valLb,0.01);
         objs(i) = objVal;
         aps(i) = save(w,b);
    end
        
    plot(aps)
    plot(objs)

  
end

function ap = save(w,b)
    
    HW2_Utils.genRsltFile(w,b,"val","out");
    ap = HW2_Utils.cmpAP("out","val");
end

function [w,b,alpha,obj] = svm_qp(X, y, valD, valLb,  c)
    [N,~] = size(y);
    H=(y*y').*(X'*X);
    f=-ones(1,N);
    A=zeros(1,N);
    Aeq = y';
    b = 0;
    beq = 0;
    cval = 0;
    LB=zeros(N,1);
    UB = c*ones(N,1);        
    

    alpha = quadprog(H,f,A,b,Aeq,beq,LB,UB);

    w = X * (alpha.*y);
    
    j=min(find(alpha>0&y==1));  
    kernal = X'*X;
    b = 1 - kernal(j,:)*(alpha.*y);
    
    obj = 1/2*sum(w.*w) + c*(sum(max(1-valLb.*(valD'*w + b ),0) ));
    
end

function hog_vectors = mine(w,b,ds)
    eps = 92;
    
    hog_vectors = [];
    
    x=load('trainAnno.mat');
    
    for i=1:eps
        im = sprintf('%s/%s/%04d.jpg', HW2_Utils.dataDir,ds,i);% mod(i,93)+1);
        im = imread(im);
        rects = HW2_Utils.detect(im,w,b,false);
        rects = rects(1:5,:) ;
        [n,~] = size(rects);
        r = x.ubAnno{i};%mod(i,93)+1};
               

        for j=1:n
            overlap = HW2_Utils.rectOverlap(r,rects(1:4,j));
            if overlap < 0.5
                "overlap less than 0.3" ;
                svm_score = rects(5,j);
                
                if svm_score>0
                    
                    rects(:,j);
                    left = rects(1,j);
                    top = rects(2,j);
                    right = rects(3,j);
                    bottom= rects(4,j);

                    
                    
                    if left>640 | top>360 | right>640 | bottom>360
                        'wasted rectangle';
                        continue;
                    end
                    
                    imReg = im(top:bottom, left:right,:);
                    size(imReg);
                    imReg = imresize(imReg, HW2_Utils.normImSz);
                    v = HW2_Utils.cmpFeat(rgb2gray(imReg));                    
                    hog = vl_hog(im2single(imReg), HW2_Utils.hogCellSz) ;
                    perm = vl_hog('permutation') ;
                    flippedHog = hog(:,end:-1:1,perm) ;
                    flippedHog = flippedHog(:);
                    
                    hog_vectors = cat(2,hog_vectors,flippedHog);
                    hog_vectors = cat(2,hog_vectors,v);
                    hog_vectors = cast(hog_vectors,'double');
                    
                end
         
            end
        end
        
    end
end

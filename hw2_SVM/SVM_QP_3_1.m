load '/Users/hiren/Documents/MATLAB/hw2data/q3_1_data.mat';
'c=0.1'

c = 0.1;
[w,b,alpha,obj,valAcc] = svm_qp(trD, trLb, valD, valLb,c);
valAcc
obj
size(alpha(alpha < 0.1))


'c=10'

c = 10;
[w,b,alpha,obj,valAcc] = svm_qp(trD, trLb, valD, valLb,c);
valAcc
obj
size(alpha(alpha > 0.01))

function [w,b,alpha,obj,valAcc] = svm_qp(X, y, valD, valLb,  c)
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

    valAcc = predictEval(w,b, valD, valLb);
    
end

function valAcc = predictEval(w,b, valD, valY)
    [nVal,~] = size(valY);
    yPred = valD'*w + b*ones(nVal,1);
    
    yPred(yPred<1) = -1;
    yPred(yPred>=1) = 1;    
    yPred = yPred';
    correct = 0 ;

    'Confusion Matrix'
    [C,order] = confusionmat(valY,yPred)
    
    for j = 1:nVal
        if valY(j) == yPred(j)
            correct = correct + 1;
        end
    end

    valAcc = correct/nVal*100;

end


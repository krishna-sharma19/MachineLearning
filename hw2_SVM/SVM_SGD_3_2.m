q_3_7()

function q_3_5()

    load '/Users/hiren/Documents/MATLAB/hw2data/q3_1_data.mat';
    epochs = 1999;
    n0=1;
    n1=100;
    c = 0.1;
    k=7;
    X = trD;
    y = trLb;
    %X = horzcat(trD,valD);
    %y = vertcat(trLb,valLb);

    [w, valAcc, loss] = svm_sdg(X,y, epochs, n0, n1, c, valD, valLb);
    
    c = 10;
%    [w, valAcc, loss] = svm_sdg(X,y, epochs, n0, n1, c, valD, valLb);
    valAcc
    loss
    k = size(unique(y));
    k = k(1);
    temp = 0
    for j =1:k
        temp = temp + norm(w(:,j))*norm(w(:,j));
    end
    temp
end

function q_3_7()

    load '/Users/hiren/Documents/MATLAB/hw2data/q3_2_data.mat';
    X = trD;
    y = trLb;
    epochs = 100;
    n0=1;
    n1=100;
    c = 0.0008;
    k=7;
    %X = horzcat(trD,valD);
    %y = vertcat(trLb,valLb);
    %X = trD;
    %y = trLb;

    [w, valAcc, loss] = svm_sdg(X,y, epochs, n0, n1, c, valD, valLb);

    yPred = predictWrite(w, tstD);
end


function [w, valAcc, loss] = svm_sdg(X,y, epochs, n0, n1, c,valX, valY)
    y(y == -1) = 2;

    [d, n] = size(X);

    pValAcc = 0;
    valAcc = 1;
    epoch = 0;
    k = size(unique(y));
    k = k(1);
    w = zeros(d,k);
    lossPlot = zeros(epochs+1,1);
    
%    while pValAcc ~= valAcc
    for epoch =1:epochs
        epoch = epoch+1;
        eta = n0/(n1+2*epoch);
        randArr = randperm(n);

        for i = randArr
            maxE = -Inf;
            yicap = -1;
            yi = y(i);


            for j = 1:k
                currM = w(:,j)' * X(:,i);

                if currM > maxE && j ~= yi
                    yicap = j;
                    maxE = currM;
                end            
            end        
            tempL = 0;
            loss = max(w(:,yicap)' * X(:,i) - w(:,yi)' * X(:,i) + 1, 0);
            for j = 1:k
                delLi = NaN;
                tempL = tempL +  norm(w(:,j))*norm(w(:,j));
                l = zeros(d,1);
                if loss >0
                    l = c*X(:,i);
                end

                if j == yi
                    delLi = w(:,j)/n - l;

                elseif j==yicap
                    delLi = w(:,j)/n + l;

                else
                    delLi = sum(w,2)/n;
                end

                w(:,j) = w(:,j) - eta*delLi; 
            end
            
            lossPlot(epoch) = lossPlot(epoch) + c*loss + tempL/(2*n);
        end
        pValAcc = valAcc;
        epoch;
        valAcc = predictEval(w, valX, valY);
        
    end
    plot(lossPlot)
end

function valAcc = predictEval(w, valX, valY)
    [d1, nVal] = size(valX);

    [~, yPred] = maxk(w'*valX , 1);

    valY(valY == -1) = 2;

    yPred = yPred';
    correct = 0 ;

    for j = 1:nVal
        if valY(j) == yPred(j)
            correct = correct + 1;
        end
    end

    valAcc = correct/nVal*100;

end

function kfold()
    load fisheriris 
    indices = crossvalind('Kfold',y,k);
    valAcc = 0;

    for i = 1:k
        test = (indices == i); 
        train = ~test;

        tX = X(:,train);
        vX = X(:,test);

        ty = y(train,:);
        vy = y(test,:);

        w = svm_sdg(tX,ty, epochs, n0, n1, c, vX, vy);
        valAcc = valAcc + predictEval(w, valD, valLb);

    end

    valAcc/k
end

function yPred = predictWrite(w, valX)
    [d1, nVal] = size(valX);

    [~, yPred] = maxk(w'*valX , 1);
    yPred = yPred';
    csvwrite('HW2ActivityRecognition2.csv', yPred)
end

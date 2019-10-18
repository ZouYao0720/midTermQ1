clear;
clc;
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(3,N);
figure(1),clf, colorList = 'rbg';
ActuallabelNum = zeros(1,3);  %%% recode the acutual number of each class
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(1:2,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    x(3,indices) = l;
    ActuallabelNum(1,l) = length(indices);
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
%%%show the num of each label
label1 = sprintf('%d number for class1',ActuallabelNum(1));
label2 = sprintf('%d number for class1',ActuallabelNum(2));
label3 = sprintf('%d number for class1',ActuallabelNum(3));
legend(label1,label2,label3,'FontSize',16)

EstiMu = m;
EstSigma = Sigma;

%%% estimate every sample for their lable and store the prediction lables
%%% in the x(4)
error = 0;

%%this is the predict processing
for n = 1:N
    s11 = -1/2* (x(1:2,n)-EstiMu(:,1))'*(EstSigma(:,:,1))^(-1)*(x(1:2,n)-EstiMu(:,1))+log(classPriors(1))-1/2*log(det(EstSigma(:,:,1)))/pi;
    s12 = -1/2* (x(1:2,n)-EstiMu(:,2))'*(EstSigma(:,:,2))^(-1)*(x(1:2,n)-EstiMu(:,2))+log(classPriors(2))-1/2*log(det(EstSigma(:,:,2)))/pi;
    s13 = -1/2* (x(1:2,n)-EstiMu(:,3))'*(EstSigma(:,:,3))^(-1)*(x(1:2,n)-EstiMu(:,3))+log(classPriors(3))-1/2*log(det(EstSigma(:,:,3)))/pi;
        
    if s11 > s12 && s11>s13
       x(4,n) = 1;       %%%assign label 1
    elseif s12 > s11 && s12>s13
       x(4,n) =2;        %%%assign label 2
    elseif s13>s11&&s13>s12
        x(4,n) = 3;      %%%assign label 3
    end
    
    if x(4,n) ~= x(3,n)
        error = error+1;   %%% calculate the total misclassified sample
    end
end
r = x(4,:);  %%%% predict label
c = x(3,:);  %%%% true label

%%% get the confusion mat
ConfuMa = confusionmat(r,c);
figure(2)
confusionchart(ConfuMa);

%%%% plot the predict boundary
ind1MAP = find(x(3,:) == 1&(x(4,:) == 2 | x(4,:) == 3));
ind2MAP = find(x(3,:) == 2&(x(4,:) == 1 | x(4,:) == 3));
ind3MAP = find(x(3,:) == 3&(x(4,:) == 2 | x(4,:) == 1));
misclassif1 = x(1:2,ind1MAP);
misclassif2 = x(1:2,ind2MAP);
misclassif3 = x(1:2,ind3MAP);
%%% generate the total Num of miss and probabilty
totalNummiss = sprintf('%d missclassified',error);
errorPorbability = sprintf('%.3f percent error',error/sum(ActuallabelNum)*100); 
figure(3)
gscatter(x(1,:),x(2,:),x(3,:),'g','+o*'); hold on
scatter(misclassif1(1,:),misclassif1(2,:),'m','+');hold on
scatter(misclassif2(1,:),misclassif2(2,:),'m','o');hold on
scatter(misclassif3(1,:),misclassif3(2,:),'m','*');
legend('sample class1','sample class2','sample class3','for missclassified class1','for missclassified class2','for missclassified class3','FontSize',16);
%% show the missclassified num and probability
text(1,2.3,totalNummiss,'FontSize',16)
text(1,2,errorPorbability,'FontSize',16)

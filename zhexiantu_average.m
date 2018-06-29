close all
clear
clc

n = 6;
k = 1:n;
x = 2.^k;



y1 = [0.9547    0.8255  0.6882  0.5390  0.4288  0.3499];  % LLP-ELM 
y2 = [0.6318    0.3819  0.2597  0.1635  0.1273  0.1299];  % alter-svm 
y3 = [0.7066    0.6803  0.5781  0.4594  0.3633  0.3171];  % Inverse
%y4 = [0.5396	0.4739	0.4344	0.4959	0.5586	0.4982];  % con-svm 


% 图像绘制
figure;
%plot(k,y1,'-^r', k,y2,'-og', k,y3,'-db', k,y4,'-pc','LineWidth',1.8);
plot(k,y1,'-^r', k,y2,'-og', k,y3,'-db', 'LineWidth',1.8);
ylabel('mean accuracy');
xlabel('Bag Size');
legend('LLP-ELM','alter-SVM','InvCal');


set(gca,'XTick',k);
set(gca,'XTickLabel',{'2','4','8','16','32','64'});
set(gca,'fontsize',12);
set(gca,'YLim',[0.1 0.99])
%set(handles,'ytick',0.4:0.1:0.9) % handles可以指定具体坐标轴的句柄

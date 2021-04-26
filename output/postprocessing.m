close all
clear all

bigData = [];
axisx = 10:10:40;
for ii = axisx
    text = ['output_error_wval' num2str(ii) '.txt'];
    data = load(text);
    for jj = 1:length(data)
        %Assuming Neural Network Net trained with maximum loss of 40%
        if data(jj)> 0.4
            data(jj) = 0;
        end
    end
    bigData = [bigData mean(data)];
end

plot(axisx,bigData, '-k^','LineWidth',1.5)
%legend('10','15','20','25','30','35','40','45','50')
xticks([10 20 30 40])
xticklabels({'10','20','30','40'})
yticks([ 0.0224 0.0448 0.0874 0.1218  ])
%yticklabels({'10','20','30','40'})
xlabel('Forecasting Windows Size (samples)','FontSize',14)
ylabel('False Alarm Probability','FontSize',14)


set(gca,'fontsize',10,'FontWeight','bold')
grid on
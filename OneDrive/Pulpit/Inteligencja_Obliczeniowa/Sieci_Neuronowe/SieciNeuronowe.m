clear all;
close all;

%% parametry
a = 0;        
b = 12;
ile_epok_list = [50, 100, 500];                % liczby epok 
ile_danych_list = [20, 50, 100, 300, 1000];     % różne liczby danych uczących
neuron_configurations = {
    [1], [4], [8], [20], [40], [20 20]
};

%% funkcja
x = a:0.05:b;
y = 0.5*cos(0.2.*x.^2) + 0.5;


for epoki = ile_epok_list
    for ile_danych = ile_danych_list
        for i = 1:length(neuron_configurations)
            netconf = neuron_configurations{i};

            % dane uczace z szumem
            x_siec = a + (b - a) * rand(ile_danych, 1);
            y_siec = 0.5 * cos(0.2 * x_siec.^2) + 0.5 + 0.1 * randn(ile_danych, 1);

            % siec
            net = feedforwardnet(netconf);
            net.trainParam.epochs = epoki;
            net.trainParam.goal = 0;
            net.trainParam.max_fail = epoki; % pełny przebieg epok

            % uczenie
            net = train(net, x_siec.', y_siec.');

            
            ypred = net(x);

            
            figure('Visible','off'); 
            hold on;
            plot(x, y, 'k', 'LineWidth', 2);                    
            plot(x_siec, y_siec, 'rx', 'MarkerSize', 5);        
            plot(x, ypred, 'b', 'LineWidth', 2);                
            title(sprintf('Neurony: [%s], Epoki: %d, Przypadki: %d', ...
                num2str(netconf), epoki, ile_danych));
            legend('Funkcja docelowa','Dane uczące','Aproksymacja sieci');
            xlabel('x'); ylabel('y');

            % zapisujemy wykresy 
            nazwa = sprintf('wykres_N%s_E%d_D%d.png', ...
                strrep(num2str(netconf), ' ', '-'), epoki, ile_danych);
            saveas(gcf, fullfile('Wyniki', nazwa));
            close;
        end
    end
end


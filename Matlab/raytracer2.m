clearvars; close all;

testRx = [2, 3, 0];%x,y,0

testPeople = [1, 2, 3, 4, 0]; %x,y,w,h,0

txPos = [2.5, 2.5];
rxPos = [0.1, 0.1, 0;...
         0.1, 2.5, 0;...
         0.1, 4.9, 0;...
         2.5, 4.9, 0;...
         4.9, 4.9, 0;...
         4.9, 2.5, 0;...
         4.9, 0.1, 0;...
         2.5, 0.1, 0];

%peoplePos = [3, 1, 1, 1, 0];

roomHeight = 5;
roomWidth = 5;
numRefl = 3;
reflCoeff = 0.5;
wavelength = 0.12;

runs = 12;
RSSI = zeros(runs, size(rxPos,1));
pepolePosMatrix = zeros(runs, 5);

for run = 1:runs
    peoplePos = [rand() * roomWidth, rand() * roomHeight, 1, 1, 0];
    pepolePosMatrix(run,:) = peoplePos;
    figure;
    hold on;
    plot([0, 0, roomWidth, roomWidth, 0], [0, roomHeight, roomHeight, 0, 0]);
    plot(txPos(1), txPos(2), "x", "Color", "r");
    plot(rxPos(:, 1), rxPos(:, 2), "x", "Color", "b");
    plot(peoplePos(1), peoplePos(2), "o", "Color", "g");
    hold off;
    xlim([-0.25 roomWidth + 0.25]);
    ylim([-0.25 roomHeight + 0.25]);

    for i = 1:size(rxPos, 1)
        RSSI(run,i) = calcLoss(txPos, rxPos(i, :), peoplePos, ...
            roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
        disp("RSSI" + i + ": " + RSSI(i));
    end
    disp("------------------");

end
save('results.mat');
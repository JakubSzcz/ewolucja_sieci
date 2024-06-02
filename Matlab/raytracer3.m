clearvars; close all;

nr_people = 1;
nr_samples = 20000;

params;

%
RSSI = zeros(nr_samples, length(rxPos));
positions = zeros(nr_samples, 2 * nr_people);
furniturePos = [1, 2, 1, 1, 0];


for iter = 1 : nr_samples
    clc;
    disp("people: " + iter + " / " + nr_samples);
    people = zeros(nr_people, 5);
    for p = 1 : nr_people
        people(p, :) = [rand * roomWidth, rand * roomHeight, person_width, person_width, 0];
        positions(iter, (p-1)*2 + 1 : (p-1)*2 + 2) = people(p, 1:2);
    end

    % loop through rx antennae
    for i = 1:size(rxPos, 1)
        %RSSI(iter, i) = calcLoss(txPos, rxPos(i, :), people, ...
            %roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
        RSSI(iter, i) = calcLoss(txPos, rxPos(i, :), people, ...
            furniturePos, roomHeight, roomWidth, numRefl, reflCoeff, ...
            0.9, 0.1, wavelength);
    end
end

save("people.mat", "RSSI", "positions");
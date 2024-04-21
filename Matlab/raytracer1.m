clearvars; close all;

params;

peoplePos = [3, 2, person_width, person_width, 0;...
             3, 1,person_width, person_width, 0;...
             2.2, 2.5, person_width, person_width, 0];

for x = 1:roomWidth * 10
    for y = 1:roomHeight * 10
        map(y, x) = calcLoss(txPos, [x/10, y/10, 0], peoplePos, ...
            roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
    end
end
figure;
hold on;
imagesc(0:0.1:roomWidth, 0:0.1:roomHeight, log(abs(map)));
plot([0, 0, roomWidth, roomWidth, 0], [0, roomHeight, roomHeight, 0, 0]);
plot(txPos(1)-0.05, txPos(2)-0.05, "x", "Color", "r");
plot(rxPos(:, 1), rxPos(:, 2), "x", "Color", "k");
plot(peoplePos(:, 1), peoplePos(:, 2), "o", "Color", "r", "MarkerFaceColor", "r");
hold off;
xlim([-0.05 roomWidth + 0.05]);
ylim([-0.05 roomHeight + 0.05]);
xlabel("X");
ylabel("Y");

for i = 1:size(rxPos, 1)
    RSSI(i) = calcLoss(txPos, rxPos(i, :), peoplePos, ...
        roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
    disp("RSSI" + i + ": " + RSSI(i));
end
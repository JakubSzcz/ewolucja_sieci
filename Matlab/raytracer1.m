clearvars; close all;

testRx = [2, 3, 0];%x,y,0

testPeople = [1, 2, 3, 4, 0]; %x,y,w,h,0

txPos = [1, 2];
peoplePos = [3, 1, 1, 1, 0];
roomHeight = 2.5;
roomWidth = 5;
numRefl = 3;
reflCoeff = 0.5;
wavelength = 0.12;


for x = 1:50
    for y = 1:25
        map(x, y) = calcLoss(txPos, [x/10, y/10, 0], peoplePos, ...
            roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
    end
end
imagesc(0:0.1:2.5, 0:0.1:5, log(abs(map)));
xlabel("Y");
ylabel("X");

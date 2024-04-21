clearvars; close all;

people1 = false;
people2 = true;

params;

offset = 0.1;
x_ind = offset : 0.1 : roomWidth - offset;
y_ind = offset : 0.1 : roomWidth - offset;

if people1
    % 1 person
    RSSI1 = zeros(length(y_ind), length(x_ind), size(rxPos, 1));
    
    iter = 1;
    number = length(y_ind) * length(x_ind);
    for x = 1 : length(x_ind)
        clc;
        disp("people 1: " + iter + " / " + number);
        for y = 1 : length(y_ind)
            % loop through rx antennae
            for i = 1:size(rxPos, 1)
                RSSI1(y, x, i) = calcLoss(txPos, rxPos(i, :), [x_ind(x), y_ind(y), person_width, person_width, 0], ...
                    roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
            end
            iter = iter + 1;
        end
    end
save("people1.mat", "RSSI1", "y_ind", "x_ind");
end

if people2
    % 2 people
    RSSI2 = zeros(length(y_ind), length(x_ind), ...
        length(y_ind), length(x_ind), size(rxPos, 1));

    iter = 1;
    number = (length(y_ind) * length(x_ind))^2;
    
    for x1 = 1 : length(x_ind)
        for y1 = 1 : length(y_ind)
            for x2 = 1 : length(x_ind)
                clc;
                disp("people 2: " + iter + " / " + number);
                for y2 = 1 : length(y_ind)
                    % loop through rx antennae
                    for i = 1:size(rxPos, 1)
                        people = [x_ind(x1), y_ind(y1), person_width, person_width, 0;...
                                  x_ind(x2), y_ind(y2), person_width, person_width, 0];
                        RSSI2(y1, x1, y2, x2, i) = calcLoss(txPos, rxPos(i, :), people, ...
                            roomHeight, roomWidth, numRefl, reflCoeff, wavelength);
                    end
                    iter = iter + 1;
                end
            end
        end
    end
    save("people1.mat", "RSSI2", "y_ind", "x_ind");
end
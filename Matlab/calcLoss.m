function signal = calcLoss(txPos, rxPos, peoplePos, roomHeight, roomWidth, numRefl, reflCoeff, wavelength)
    signal = 0;
    for rxPoint = expandRx(rxPos, roomHeight, roomWidth, numRefl)' %dla kazdej sciezki
    %rxPoint = rxPos;
        rxPoint = rxPoint';
        sumObstacles = 0;
         for person = expandPeople(peoplePos, roomHeight, roomWidth, numRefl)' %dla każdego odbicia sprawdź czy przecina ono daną ścieżkę
             person = person';
             %person = peoplePos;
             if wektorsektor(txPos(1), txPos(2), rxPoint(1), rxPoint(2), person(1), person(2), person(3), person(4)) >= 0
                 sumObstacles = sumObstacles + 1;
             end
         end
        dist = sqrt((txPos(1) - rxPoint(1)).^2 + (txPos(2) - rxPoint(2)).^2);
        phi = mod(dist, wavelength)*2*pi;
        loss = 1/dist;%(wavelength/(4*pi*dist)).^2;
        if sumObstacles > 0
            loss = 0;
        end
        signal = signal + loss*exp(i*phi);
     end
 end

function expandedTotal = expandRx(rxPos, roomHeight, roomWidth, numRefl)
    expandedUp = moveOneUpRx(rxPos, rxPos, roomHeight, numRefl);

    expandedDown = expandedUp;
    expandedDown(:, 2) = expandedDown(:,2) .* -1; %odbicie względem osx
    
    expandedVertically = cat(1, expandedUp, expandedDown);
    
    expandedRight = moveOneRightRx(expandedVertically, expandedVertically, roomWidth, numRefl);
    
    expandedLeft = expandedRight;
    expandedLeft(:, 1) = expandedLeft(:,1) .* -1;
    
    expandedTotal = cat(1, expandedRight, expandedLeft);
    % scatter(expandedTotal(:,1), expandedTotal(:,2))


end

function expandedTotal = expandPeople(peoplePos, roomHeight, roomWidth, numRefl)
    expandedUp = moveOneUpPeople(peoplePos, peoplePos, roomHeight, numRefl);

    expandedDown = expandedUp;
    expandedDown(:, 2) = expandedDown(:,2) .* -1; %odbicie względem osx
    expandedDown(:, 4) = expandedDown(:,4) .* -1;

    expandedVertically = cat(1, expandedUp, expandedDown);
    
    expandedRight = moveOneRightPeople(expandedVertically, expandedVertically, roomWidth, numRefl);
    
    expandedLeft = expandedRight;
    expandedLeft(:, 1) = expandedLeft(:,1) .* -1;%odbicie względem osy
    expandedLeft(:, 3) = expandedLeft(:,3) .* -1;
    
    expandedTotal = cat(1, expandedRight, expandedLeft);
    %scatter(expandedTotal(:,1), expandedTotal(:,2))


end

function finalArray = moveOneUpRx(originalArray, firstArray, roomHeight, numRefl)
    %newArray = originalArray;
    newArray = [];
    for idx = 1:length(firstArray(:, 1))%for each point in the current room
        point = originalArray(idx, :);
        if point(3) < numRefl %if reflections are left
            if mod(point(3), 2) == 1 %If point is odd
                newArray = cat(1, newArray, ([point(1), point(2)+2*firstArray(idx, 2), point(3)+1]));
            else %if point is odd
                newArray = cat(1, newArray, ([point(1), point(2)+2*(roomHeight - firstArray(idx, 2)), point(3)+1]));
            end
        end
    end
    if newArray(end, end) < numRefl %if there are reflections left
        finalArray = cat(1, originalArray, moveOneUpRx(newArray, firstArray, roomHeight, numRefl));
    else
        finalArray = cat(1, originalArray, newArray);
    end
end

function finalArray = moveOneRightRx(originalArray, firstArray, roomWidth, numRefl)
    %newArray = originalArray;
    newArray = [];
    for idx = 1:length(firstArray(:, 1))%for each point in the current room
        point = originalArray(idx, :);
        if point(end) < numRefl %if reflections are left
            if mod(point(end), 2) == 1 %If point is odd
                newArray = cat(1, newArray, ([point(1)+2*firstArray(idx, 1), point(2), point(3)+1]));
            else %if point is odd
                newArray = cat(1, newArray, ([point(1)+2*(roomWidth - firstArray(idx, 1)), point(2), point(3)+1]));
            end
        end
    end
    if newArray(end, end) < numRefl %if there are reflections left
        finalArray = cat(1, originalArray, moveOneRightRx(newArray, firstArray, roomWidth, numRefl));
    else
        finalArray = cat(1, originalArray, newArray);
    end
end

function finalArray = moveOneUpPeople(originalArray, firstArray, roomHeight, numRefl)
    %newArray = originalArray;
    newArray = [];
    for idx = 1:length(firstArray(:, 1))%for each point in the current room
        point = originalArray(idx, :);
        if point(end) < numRefl %if reflections are left
            if mod(point(end), 2) == 1 %If point is odd
                newArray = cat(1, newArray, ([point(1), point(2)+2*firstArray(idx, 2), point(3), point(4).*-1, point(5)+1]));
            else %if point is odd
                newArray = cat(1, newArray, ([point(1), point(2)+2*(roomHeight - firstArray(idx, 2)), point(3), point(4).*-1, point(5)+1]));
            end
        end
    end
    if newArray(end, end) < numRefl %if there are reflections left
        finalArray = cat(1, originalArray, moveOneUpPeople(newArray, firstArray, roomHeight, numRefl));
    else
        finalArray = cat(1, originalArray, newArray);
    end
end

function finalArray = moveOneRightPeople(originalArray, firstArray, roomWidth, numRefl)
    %newArray = originalArray;
    newArray = [];
    for idx = 1:length(firstArray(:, 1))%for each point in the current room
        point = originalArray(idx, :);
        if point(end) < numRefl %if reflections are left
            if mod(point(end), 2) == 1 %If point is odd
                newArray = cat(1, newArray, ([point(1)+2*firstArray(idx, 1), point(2), point(3).*-1, point(4), point(5)+1]));
            else %if point is odd
                newArray = cat(1, newArray, ([point(1)+2*(roomWidth - firstArray(idx, 1)), point(2), point(3).*-1, point(4), point(5)+1]));
            end
        end
    end
    if newArray(end, end) < numRefl %if there are reflections left
        finalArray = cat(1, originalArray, moveOneRightPeople(newArray, firstArray, roomWidth, numRefl));
    else
        finalArray = cat(1, originalArray, newArray);
    end
end
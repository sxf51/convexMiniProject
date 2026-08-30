%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Task 3 Question 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% This example demonstrates vehicle path following along the center %%%%
%%%%                     path of a circular track.                     %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Create the Driving Scenario %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Automated Driving Toolbox is needed
scenario = drivingScenario();
x1_wp = [15 25 -15 -15 -15 15];
x2_wp = [0.1 25 25 -0.1 -25 0.1];
roadCenters = [x1_wp(:), x2_wp(:)];
roadWidth = 15;  
pathWidth = 10;  % consider the width of vehicles
road(scenario, roadCenters, pathWidth);
rb = roadBoundaries(scenario);
inner = flip(rb{1,2}(1:end,1:2));
inner = circshift(inner, [1, 0]);  % align the initial point
inner(1,:) = inner(end,:);
inner = inner(1:2:end,:);
outer = rb{1,1}(1:end,1:2);
outer = outer(1:2:end,:);
center = 0.5*(inner + outer);
scenario = drivingScenario();
road(scenario, roadCenters, roadWidth);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% MPC problem Initialisation %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change the following parameters if needed
ns = 4;  % number of states
ni = 2;  % number of inputs
Ts = 2.01;  % sampling time
T = size(center,1);  % number of sampling
% System matrices
A = [1 0 Ts 0; 0 1 0 Ts; 0 0 1 0; 0 0 0 1];
B = [Ts^2/2 0; 0 Ts^2/2; Ts 0; 0 Ts];
% Objective matrices
Q = diag([1, 1, 0, 0]);
R = 2.2e3 * eye(ni);
% Controller constraints
uMin = [-1; -1];
uMax = [1; 1];
% States reference
XRef = [center'; (circshift(center(:,1)', [0, -1]) - center(:,1)')/Ts; (circshift(center(:,2)', [0, -1]) - center(:,2)')/Ts];
XRef = reshape(XRef, ns*T, 1);
% States constraints
k = -(outer(:,1)-inner(:,1))./(outer(:,2)-inner(:,2));
c = [-k, ones(T,1), zeros(T,2)];
dMin = min(inner(:,2) - k.*inner(:,1),outer(:,2) - k.*outer(:,1));
dMax = max(inner(:,2) - k.*inner(:,1),outer(:,2) - k.*outer(:,1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Solving MPC problem online %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization Toolbox is needed
%*************************************************************************%
%*************************** Complete the code ***************************%
%*************************************************************************%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1 = X(1,:);
x2 = X(2,:);
v1 = X(3,:);
v2 = X(4,:);
v = v1.^2 + v2.^2;
trajectoryPoints = trajectoryGenerater(scenario, inner, outer, center, x1, x2, v);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement your QP solver
%*************************************************************************%
%*************************** Complete the code ***************************%
%*************************************************************************%


function trajectoryPoints = trajectoryGenerater(scenario, inner, outer, ref, x1, x2, v)
    % Add an ego vehicle to the scenario and set its position
    car = vehicle(scenario,'ClassID',1, ...
        'Length',3,'Width',2);
    
    % Set the trajectory for the vehicle to follow
    trajectoryPoints = [x1(:), x2(:)];
    trajectorySpeeds = v(:);
    
    trajectory(car, trajectoryPoints, trajectorySpeeds);
    
    % Plot the scenario
    plot(scenario)
    hold on
    plot(inner(:,1), inner(:,2), 'LineWidth', 1, 'Color',[0 0.8 0.5]);
    plot(outer(:,1), outer(:,2), 'LineWidth', 1, 'Color',[0 0.8 0.5]);  % Plot the margin
    plot(ref(:,1), ref(:,2), 'o', 'MarkerEdgeColor', 'red', 'MarkerSize', 4);
    plot(x1, x2, 'o', 'MarkerEdgeColor', 'blue', 'MarkerSize', 2);
    plot(x1, x2, 'blue')
    hold off
    
    while advance(scenario)
    end
end









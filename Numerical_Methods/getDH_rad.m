% Retrieve the DH table of a robotic manipulator.
% Example: DH = getDH(robot, Q_initial)
% Inputs:  robot = a string representing the robot to load
%          Q_initial = a vector representing the initial joint
%          configuration of the robot to load
% Outputs: DH = a matrix representing the corresponding DH table

function DH = getDH_rad(robot, Q_initial, unit_chosen)

    
    t = Q_initial;
    
    if (strcmp(robot, '6DoF-6R-Jaco'))

        DH = [-t(1),                 0.2755,        0,         pi/2;
               t(2)-(pi/2),          0.0,           0.41,      pi;
               t(3)+(pi/2),         -0.0098,        0,         pi/2;
               t(4),                -0.2502,        0,         pi/3;
               t(5)-(pi),           -0.08579,       0,         pi/3;
               t(6)+deg2rad(-100),  -0.2116,        0,         pi];

    elseif (strcmp(robot, '6DoF-6R-Puma560'))

        DH = [t(1),         0,           0,       -pi/2;
              t(2),   0.14909,      0.4318,           0;
              t(3),         0,     -0.0203,        pi/2;
              t(4),   0.43307,           0,       -pi/2;
              t(5),         0,           0,        pi/2;
              t(6),   0.05625,           0,           0];

    elseif (strcmp(robot, '6DoF-6R-Mico'))

        DH = [          -t(1),    0.2755,        0,         pi/2;
                  t(2)-(pi/2),       0.0,     0.29,         pi;
                  t(3)+(pi/2),    -0.007,        0,         pi/2;
                         t(4),   -0.1661,        0,         pi/3;
                    t(5)-(pi),  -0.08556,        0,         pi/3;
           t(6)+deg2rad(-100),   -0.2028,        0,         pi];


    elseif (strcmp(robot, '6DoF-6R-IRB140'))
        
        DH = [t(1),     0.352,     0.07,        -pi/2;
              t(2),         0,     0.36,         0;
              t(3),         0,        0,        -pi/2;
              t(4),      0.38,        0,         pi/2;
              t(5),         0,        0,        -pi/2;
              t(6),     0.065,        0,         0];


    elseif (strcmp(robot, '6DoF-6R-KR5'))    
        
        DH = [t(1),     0.4,    0.18,     -pi/2;
              t(2),     0,      0.6,       0;
              t(3),     0,      0.12,      pi/2;
              t(4),    -0.62,   0,        -pi/2;
              t(5),     0,      0,         pi/2;
              t(6),    -0.115,  0,         pi];

    elseif (strcmp(robot, '6DoF-6R-UR10'))
    
        DH = [t(1),  0.1273,    0,          pi/2;
              t(2),       0,    -0.612,     0;
              t(3),       0,    -0.5723,    0;
              t(4),  0.1639,    0,          pi/2;
              t(5),  0.1157,    0,         -pi/2;
              t(6),  0.0922,    0,         0];

    elseif (strcmp(robot, '6DoF-6R-UR3'))

        DH = [t(1),  0.1519,    0,              pi/2;
              t(2),       0,    -0.2437,        0;
              t(3),       0,    -0.2132,        0;
              t(4),  0.1124,    0,              pi/2;
              t(5), 0.08535,    0,             -pi/2;
              t(6),  0.0819,    0,              0];
        
    elseif (strcmp(robot, '6DoF-6R-UR5'))

        DH = [t(1),  0.08946,       0,          pi/2;
              t(2),        0,   0.425,          0;
              t(3),        0, -0.3922,          0;
              t(4),   0.1091,       0,          pi/2;
              t(5),  0.09465,       0,         -pi/2;
              t(6),   0.0823,       0,          0];

    elseif (strcmp(robot, '6DoF-6R-Puma260'))    
        
        DH = [t(1),         0,           0,         -pi/2;
              t(2),         0.1254,      0.2032,     0;
              t(3),         0,          -0.0079,     pi/2;
              t(4),         0.2032,      0,         -pi/2;
              t(5),         0,           0,          pi/2;
              t(6),         0.0635,      0,          0];

    elseif (strcmp(robot, '6DoF-2RP3R-Stanford'))
    
        DH = [t(1),     0.412,      0.0,    -pi/2;
            t(2),       0.154,      0.0,     pi/2;
            -90.0,      t(3),       0.0203,  0.0;
            t(4),       0.0,        0.0,    -pi/2;
            t(5),       0.0,        0.0,     pi/2;
            t(6),       0.0,        0.0,     0.0];



    elseif (strcmp(robot, '7DoF-7R-Jaco2'))   
        
        DH = [t(1)+pi,      -0.2755,    0,      pi/2;
             t(2),          0.0,        0,      pi/2;
             t(3),          -0.41,      0,      pi/2;
             t(4),          -0.0098,    0,      pi/2;
             t(5),          -0.3111,    0,      pi/2;
             t(6),          0.0,        0,      pi/2;
             t(7)+pi/2,     -0.2638,    0,        pi];

        



    elseif (strcmp(robot, '7DoF-7R-Panda'))

        DH = [t(1),   0.333,     0.0,        0;
              t(2),     0.0,     0.0,  -pi/2;
              t(3),   0.316,     0.0,   pi/2;
              t(4),     0.0,  0.0825,   pi/2;
              t(5),   0.384, -0.0825,  -pi/2;
              t(6),     0.0,     0.0,   pi/2;
              t(7),   0.107,   0.088,   pi/2];

    elseif (strcmp(robot, '7DoF-7R-WAM'))

        DH = [t(1),     0.0,    0.0,  -pi/2;
              t(2),     0.0,    0.0,   pi/2;
              t(3),   0.550,    0.0,  -pi/2;
              t(4),     0.0,  0.045,   pi/2;
              t(5),   0.300, -0.045,  -pi/2;
              t(6),     0.0,    0.0,   pi/2;
              t(7),   0.060,    0.0,      0];

    elseif (strcmp(robot, '7DoF-7R-Baxter'))

       DH = [t(1),          0.27,       0.069,     -pi/2;
             t(2)+pi/2,     0.0,        0.0,        pi/2;
             t(3),          0.364,      0.069,     -pi/2;
             t(4),          0.0,        0.0,        pi/2;
             t(5),          0.374,      0.01,      -pi/2;
             t(6),          0.0,        0.0,        pi/2;
             t(7),          0.28,       0.0,        0];
    elseif (strcmp(robot, '7DoF-7R-Sawyer'))

        DH = [t(1),    0.317,   0.081,   -pi/2;
              t(2),    0.1925,  0.0,     -pi/2;
              t(3),    0.4,     0.0,     -pi/2;
              t(4),    0.1685,  0.0,     -pi/2;
              t(5),    0.4,     0.0,     -pi/2;
              t(6),    0.1363,  0.0,     -pi/2;
              t(7),    0.1338,  0.0,      0.0];

    elseif (strcmp(robot, '7DoF-7R-KukaLWR4+'))

         DH = [t(1),    0.3105,  0.0,   pi/2;
              t(2),    0.0,     0.0,  -pi/2;
              t(3),    0.4,     0.0,  -pi/2;
              t(4),    0.0,     0.0,   pi/2;
              t(5),    0.39,    0.0,   pi/2;
              t(6),    0.0,     0.0,  -pi/2;
              t(7),    0.078,   0.0,   0.0];

    elseif (strcmp(robot, '7DoF-7R-PR2Arm'))

        DH = [t(1),    0.333,  0.0,   -pi/2;
              t(2),    0.0,    0.350,  pi/2;
              t(3),    0.0,    0.400,  pi/2;
              t(4),    0.4,    0.0,   -pi/2;
              t(5),    0.0,    0.0,    pi/2;
              t(6),    0.0,    0.0,   -pi/2;
              t(7),    0.082,  0.0,    0.0];

    elseif (strcmp(robot, '7DoF-7R-PA10'))

        DH = [t(1),    0.317,  0.0, -pi/2;
              t(2),    0.0,    0.0,  pi/2;
              t(3),    0.450,  0.0, -pi/2;
              t(4),    0.0,    0.0,  pi/2;
              t(5),    0.480,  0.0, -pi/2;
              t(6),    0.0,    0.0,  pi/2;
              t(7),    0.070,  0.0,  0.0];

    elseif (strcmp(robot, '7DoF-7R-Gen3'))

        DH = [t(1),        -0.2848,  0.0,    pi/2;
              t(2) + pi,    -0.0118,  0.0,    pi/2;
              t(3) + pi,    -0.4208,  0.0,    pi/2;
              t(4) + pi,    -0.0128,  0.0,    pi/2;
              t(5) + pi,    -0.3143,  0.0,    pi/2;
              t(6) + pi,     0.0,     0.0,    pi/2;
              t(7) + pi,    -0.1674,  0.0,    pi];


    elseif (strcmp(robot, '7DoF-2RP4R-GP66+1'))

        DH = [t(1),  0.0,  0.0,    pi/2;
              t(2),  0.0,  0.25,   pi/2;
              0.0,   t(3), 0.0,    0.0;
              t(4),  0.0,  0.0,    pi/2;
              t(5),  0.14, 0.0,    pi/2;
              t(6),  0.0,  0.0,    pi/2;
              t(7),  0.0,  0.0,    pi/2];


end
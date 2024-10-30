function r = getRobotConfiguration(robot_chosen, unit_chosen, DH)

    robot_list_6DoF_c = {'6DoF-6R-Jaco', '6DoF-6R-Puma560', '6DoF-6R-Mico', '6DoF-6R-IRB140', '6DoF-6R-KR5', ...
                         '6DoF-6R-UR10', '6DoF-6R-UR3', '6DoF-6R-UR5', '6DoF-6R-Puma260'};
    robot_list_6DoF_p = { '6DoF-2RP3R-Stanford'};

    robot_list_7DoF_c = {'7DoF-7R-Jaco2', '7DoF-7R-Panda', '7DoF-7R-WAM', '7DoF-7R-Baxter', '7DoF-7R-Sawyer', ...
                         '7DoF-7R-KukaLWR4+', '7DoF-7R-PR2Arm', '7DoF-7R-PA10', '7DoF-7R-Gen3'};
     
    robot_list_7DoF_p = {'7DoF-2RP4R-GP66+1'};
    
    % Check if robot_choice exists in the list
    if ismember(robot_chosen, robot_list_6DoF_c)
        
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0   DH(3, 2)    0],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6}); 

    elseif ismember(robot_chosen, robot_list_7DoF_c)
        
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0   DH(3, 2)    0],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard');
        L7 = link([DH(7, 4)    DH(7, 3)     0   DH(7, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7});
        
        
    elseif ismember(robot_chosen, robot_list_6DoF_p)
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0          0    1],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6});  

    elseif ismember(robot_chosen, robot_list_7DoF_p)
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0          0    1],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard');
        L7 = link([DH(7, 4)    DH(7, 3)     0   DH(7, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7});    

    end

end
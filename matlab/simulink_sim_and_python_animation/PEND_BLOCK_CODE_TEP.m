function Dx = dynamika_wahadla(x, u, params)
       % x : state
       % u : input signal
       %
       % params scope: "parameters"

       % State, function input: x = [x Dx theta Dtheta]'
       x1 = x(1);
       x2 = x(2);
       x3 = x(3);
       x4 = x(4);

       % Params
       (M, mc, mp, Lp, Lc, g, b, gamma, D, alpha) = params;
       mr = mc + mp;
       Mt = mr + M;
       L  = (Lc*mc + Lp*mp) / mr;
       Jcm = (L^2)*mr + (Lc^2)*mc + 4/3 * (Lp^2)*mp;
       Jt  = Jcm + mr*L^2;

       % Equations for pendulum dynamics
       den = Jt*Mt - L^2 * mr^2 * cos(x2);
       ddt_x1 = x3;
       ddt_x2 = x4;
       ddt_x3 = ( Jt*(D*cos(alpha) + L*mr*x4**2*sin(x2) - b*x3 + u) - ...
                     L*mr*cos(x2) * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) ...
                     / den;

       ddt_x4 = ( -L*mr*cos(x2) * (D*cos(alpha) + L*mr*x4^2*sin(x2) - b*x3 + u) + ...
                     Mt * ( D*( L*cos(alpha-x2) + sin(alpha+np.pi/4) ) + L*g*mr*sin(x2) - gamma*x4 ) ) ...
                     / den;

       % output
       Dx = [ddt_x1 ddt_x2 ddt_x3 ddt_x4]';
end
       
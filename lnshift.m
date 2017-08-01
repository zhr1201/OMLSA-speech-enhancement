function y = lnshift(x,t)
szX = size(x);
if szX(1) > 1
    n = szX(1);
    y = [x((1 + t):n); x(1:t)];
else
    n = szX(2);
    y = [x((1 + t):n) x(1:t)];
end


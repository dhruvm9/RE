%Load tracking data 
position = importdata('1820-211231_3_posMATLAB.csv');
t = position.data(:,1);
x = position.data(:,2);
y = position.data(:,3);

epochs = importdata('1820-211231_3_fwd.csv');
start = epochs.data(:,1); 
fin = epochs.data(:,2);

x0 = tsd(t,x);
y0 = tsd(t,y);

allidphi = [];
x1_py = {};
y1_py = {};
r_py = {};
v_py = {};

for i = 1: size(start,1)
    x1 = restrict(x0,start(i),fin(i));
    y1 = restrict(y0,start(i),fin(i));
    
    x1_py{i} = x1.D;
    y1_py{i} = y1.D; 

    [ dx ] = dxdt(x1, 'window', 1, 'postSmoothing',0.5);
    [ dy ] = dxdt(y1, 'window', 1, 'postSmoothing',0.5);
        

    phi = tsd(dx.range(), atan2(dy.data(), dx.data()));
    uphi = tsd(phi.range(), unwrap(phi.data()));
    dphi = dxdt(uphi, 'window', 1, 'postSmoothing',0.5);
    tmp = abs(dphi.D);
    idphi = sum(tmp,'omitnan');
    
        
    r = sqrt((dx.D).^2 + (dy.D).^2);
    r_py{i} = r;
    v = r.*100;
    v_py{i} = v;
    
    %map = colormap([r ones(size(r)) ones(size(r))]);
    %map = hsv2rgb(map);
    
    %figure()
    %plot(x,y,'Color',[0 0 0]+0.05*12)
    %hold on
    %for j =  1: size(x1.D,1)
    %    scatter(x1.D(j),y1.D(j),[],map(j,:),'filled','LineWidth',1.5)
    %    colorbar;
    %end 
    %drawnow;
    
    allidphi = [allidphi idphi];
    
end 

z1 = (allidphi - mean(allidphi)) / std(allidphi);
z2 = log(allidphi);





 
clear
clc
data = load('metalicnost.txt');
v = data(:,1)/1000;
m = data(:,2);

nbins = [50, 50];
[counts, centers] = hist3([v m], 'Nbins', nbins);

pdf2d = counts / trapz(centers{1}, trapz(centers{2}, counts, 2));

[Vgrid, Mgrid] = meshgrid(centers{1}, centers{2});

figure
surf(Vgrid, Mgrid, pdf2d')
shading interp
colormap('jet')
colorbar
xlabel('v')
ylabel('m')
zlabel('PDF(v,m)')
hold on

h = 0.035;
ymin = -2;
ymax = 1;
zmin = 0;
zmax = h;

% ravni
Yvec = linspace(ymin, ymax, 50);
Zvec = linspace(zmin, zmax, 2);
[Yp, Zp] = meshgrid(Yvec, Zvec);

xplanes = [5, 10, 26.3 58 120];   % ʻOumuamua i ATLAS

for x0 = xplanes
    % --- RAVAN ---
    Xp = x0 * ones(size(Yp));
    surf(Xp, Yp, Zp, 'FaceAlpha', 0.25, 'EdgeColor', 'none');

    % --- PRESEK pomoću interp2 ---
    z_profile = interp2(Vgrid, Mgrid, pdf2d', x0*ones(size(Yvec)), Yvec);  % interpolacija po x
    plot3(x0*ones(size(Yvec)), Yvec, z_profile, 'k', 'LineWidth', 2)
end

view(3)
hold off

% 2D plot preseka sa normalizacijom
figure
hold on
colors = lines(length(xplanes));

for i = 1:length(xplanes)
    x0 = xplanes(i);
    z_profile = interp2(Vgrid, Mgrid, pdf2d', x0*ones(size(centers{2})), centers{2});
    z_profile = z_profile / trapz(centers{2}, z_profile);  % normalizacija
    plot(centers{2}, z_profile, 'LineWidth', 2, 'Color', colors(i,:))
end

xlabel('m')
ylabel('Conditional PDF at x = const')
legend(arrayfun(@(x) sprintf('x = %.1f',x), xplanes, 'UniformOutput', false))
grid on
hold off




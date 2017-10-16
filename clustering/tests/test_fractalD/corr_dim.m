% toy example of correlation dimension calculation
% Ref: http://en.wikipedia.org/wiki/Correlation_dimension

D = 3;

% 11 frames of COM coordinate for 201 monomer/frame
load('mappedALL.mat');


%N = length(mappedALL);
%coords = mappedALL;
N = 198;
eps = 0.1:0.1:22;
Cadd = zeros(numel(eps),1);

%10 frames for compute average and errors of correlation dimension, each
%frame has 201 monomer.
for i = 1:10
    coords = mappedALL(201*(i-1)+3:201*(i-1)+200,:);
    fid = fopen(['coords_' int2str(i) '.dat'],'w');
    for r = 1:size(coords,1)
       fprintf(fid,'%f\t%f\t%f\n',coords(r,1),coords(r,2),coords(r,3)); 
    end
    fclose(fid);
% computing correlation integral at each dimension at each epsilon
C = nan(numel(eps),1);
%for dim=1:D

    dist = pdist(coords(:,:),'euclidean');
    for j=1:numel(eps)
 %       C(dim,i) = sum(dist<=eps(i))/N^2;
        C(j) = sum(dist<=eps(j))/N^2;
    end
    
%end

% plotting
figure;
%hold on
%for dim=1:D
%    plot(log(eps),log(C(dim,:)),'-o');
    plot(log(eps),log(C(:)),'-o');
%end
%hold off
xlabel('log(\epsilon)');
ylabel('log(C(\epsilon))');
saveas(gcf,['correlation_integral_',int2str(i)],'fig');
saveas(gcf,'correlation_integral','jpg');

Cadd = Cadd + C;


%figure;
%plot(log(eps),log(Cadd(:)),'-o');

% fitting
% - visual inspection of curves suggests linear fits of each over range eps = [exp(-2), exp(-1.5)] 
nu = nan(2,1);
%idx_lo = find(eps>exp(-2),1,'first');
%idx_hi = find(eps>exp(-1.5),1,'first');

%two regions
for dim=1:2
    
    fprintf('For dim = %d\n',dim);
    
    %mi = input('Input left edge:\n');
    %ma = input('Input right edge:\n');
    if dim == 1
        mi = -0.5;
        ma = 0.5;
    else
       mi = 1;
       ma = 2;
    end
    idx_lo = find(eps>exp(mi),1,'first');
    idx_hi = find(eps>exp(ma),1,'first');
    
    b = regress(log(C(idx_lo:idx_hi)),cat(2,log(eps(idx_lo:idx_hi))',ones(idx_hi-idx_lo+1,1)));
    nu(dim) = b(1);
end
fprintf('Corr dim estimates from linear fits over region eps = [exp(-2), exp(-1.5)]:\n');
fprintf('\n  dim corrDim\n');
for dim=1:2
    fprintf('%5d%8.2f\n',dim,nu(dim));
end
end

figure;
bar(1:1:2,nu);
xlabel('dim');
ylabel('\nu');
saveas(gcf,'correlation_dimension','fig');
saveas(gcf,'correlation_dimension','jpg');


% saving
save corr_dim.mat eps C
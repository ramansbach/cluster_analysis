for i = 1:10
   c = importdata(['cepy_' int2str(i) '.dat']);
   fig = openfig(['correlation_integral_',int2str(i)],'visible');
   hold on
   plot(log(c(:,1)),log(c(:,2)),'r.','markersize',20)
   savefig(['corrcompare_' int2str(i)])
end
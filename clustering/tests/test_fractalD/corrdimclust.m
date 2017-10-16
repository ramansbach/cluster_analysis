%go through all saved eps vs cintegral datas, plot and save by size
flist = 'datfiles';
flist = readlinebyline(flist,'string',[]);

l1 = flist{1};
f1 = readlinebyline(l1,'%f',[]);
ces = zeros(size(f1,1),length(flist)); %make array to hold correlation integrals
cess = zeros(1,length(flist)); %make array to hold aggregate sizes, each one corresponding to a column of ces
sl1 = strsplit(l1,'_');
s1 = str2double(sl1{3});
cess(1) = s1;
ces(:,1) = f1(:,2);
eps = f1(:,1);
%read in all data
for i = 2:length(flist)
   f = readlinebyline(flist{i},'%f',[]);
   l = flist{i};
   sl = strsplit(l,'_');
   s = str2double(sl{3});
   cess(i) = s;
   ces(:,i) = f(:,2);
end
sizes = unique(cess);
for i = 1:length(sizes)
   s = sizes(i);
   if s > 1 %this will always be a buncha 0s (we should probably remove from consideration)
        ce = ces(:,cess==s);
        figure()
        plot(log(eps),log(ce))
        savefig(['cdim_' int2str(s)])
        saveas(gcf,['cdim_' int2str(s)],'png')
        close
   end
end

%     for s = 1:maxats %check all possible sizes
%         figure()
%         hold on
%         saveflag = 0;
%         for ti = 1:length(ts) %check all times
%             t = ts(ti);
%         for ind = 1:maxats %check all possible indices within a size
%             cname = ['cdim_' int2str(t) '_' int2str(s) '_c_' int2str(ind) '.dat']
%             fid = fopen(cname); 
%             if fid~=-1 %check if this file exists
%                 cdi = readlinebyline(cname,'%f',[]);
%                 plot(log(cdi(:,1)),log(cdi(:,2)))
%                 saveflag = 1;
%                 fclose(fid);
%             end
%             
%         end
%         end
%         if saveflag
%         savefig(['cdim_' int2str(s)])
%         end
%         close(gcf)
%     end

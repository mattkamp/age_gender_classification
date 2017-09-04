load('wiki.mat');
% calculate age
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob); 
% concatenate age and gender and form a new matrix
sample = vertcat(age, wiki.gender);
sample = sample';
sample = num2cell(sample);
% add image path
field_fp = wiki.full_path';
% combine
wiki_mat = [field_fp, sample];
% write to file
fid = fopen('wiki.csv','wt');
fprintf(fid,'path,age,gender\n');
if fid>0
    for k=1:size(wiki_mat,1)
        fprintf(fid,'%s,%d,%d\n',wiki_mat{k,:});
    end
    fclose(fid);
end
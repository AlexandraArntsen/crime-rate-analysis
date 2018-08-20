
function [housing_data,m,n] = download_dataset() 
%------------------------------------------------%
%                                                %
%         Download Housing data from             %
%            archive.ice.uci.edu/ml/             %
%        requires internet connection            %
%                                                %
%------------------------------------------------%    
 
 repo = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/'; 
     
           url = [repo 'housing.data']; 
  housing_data = strsplit(webread(url)); 
  housing_data = housing_data(2:end-1);

  n = 14;    % number of parameters
  m = size(housing_data);
  m = m(2) / n;
  housing_data = reshape(housing_data,n,m)';
  housing_data = cellfun(@str2num,housing_data);
  
end 



function covM=safecov(covK)
        D=size(covK);
        covM=covK;

         [aaa,isPosIndef]=cholcov(covM,0);
         if isPosIndef~=0
             covM=covM+0.01*eye(D);
         end

         [aaa,isPosIndef]=cholcov(covM,0);
         if isPosIndef~=0
             covM=nearest_posdef(covM);
         end
         
        [aaa,isPosIndef]=cholcov(covM,0);
         if isPosIndef~=0
             covM=covM.*eye(D);
         end
        
end
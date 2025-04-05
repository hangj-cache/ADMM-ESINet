function  M = VariationEdge(VertConn)
nSource = size(VertConn,1);
nEdge = numel(find(VertConn(:)~=0))/2;
M = sparse(nEdge,nSource);
edge = 0;
for i = 1:nSource
    idx = find(VertConn(i,:)~=0);
    idx = idx(idx<i);
    for j = 1:numel(idx)
        M(j+edge,i) = 1;
        M(j+edge,idx(j)) = -1;     
    end
    edge = edge + numel(idx);
end
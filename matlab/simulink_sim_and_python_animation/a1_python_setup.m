echo off
if use_default_python
    pyenv;
    py_env = pyenv;
    fprintf('using: %s\n', py_env.Executable)
else
    % pyenv('Version', 'D:\minicondaaa\envs\systems\python.exe', ... 
    %       'ExecutionMode','OutOfProcess');
    pyenv('Version', 'D:\minicondaaa\envs\systems\python.exe');
    py_env = pyenv;
    fprintf('using: %s\n', py_env.Executable);
end

%
% Był problem żeby odpalić env z condy, rozwiązanie:
% https://www.mathworks.com/matlabcentral/answers/443558-matlab-crashes-when-using-conda-environment-other-than-base
%
pyExec = 'D:\minicondaaa\envs\systems\python.exe';
pyRoot = fileparts(pyExec);
p = getenv('PATH');
p = strsplit(p, ';');
addToPath = {
    pyRoot
    fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
    fullfile(pyRoot, 'Library', 'usr', 'bin')
    fullfile(pyRoot, 'Library', 'bin')
    fullfile(pyRoot, 'Scripts')
    fullfile(pyRoot, 'bin')
    };
p = [addToPath(:); p(:)];
p = unique(p, 'stable');
p = strjoin(p, ';');
setenv('PATH', p);
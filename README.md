# vim-cuda-syntax

Syntax highlighting for CUDA in Vim. Compared to Vim's default CUDA syntax file,
it adds highlighting of all CUDA defined:
- types
- enums
- constants
- global variables
- namespaces
- thrust keywords

All keywords are accumulated from the [CUDA Toolkit Documentation][toolkit].

Optionally triple-angle brackets in CUDA kernel calls can be highlighted.


## Screenshot

<dl>
<p align="center">
<img src="https://user-images.githubusercontent.com/6266600/95977810-9b746580-0e19-11eb-8105-4e2d9efa0c97.png" width="480"/>
</p>
</dl>


## Limitations

1. Highlighting of the triple angle-brackets in CUDA kernel calls works only
   when the angle brackets are on the same line. The function name is only
   highlighted when called without template arguments, i.e. `MyKernel` won't be
   highlighted in `MyKernel<foo, bar><<<grid, threads>>>(data)`.

2. CUDA [data fields][data-fields] are not highlighted because many keywords
   have familiar names which could collide with either user-defined variables
   (like `ptr`, `x`, `y`), or with C++ standard library types (like `function`
   or `array`) and would mess up the highlighting. If you want structure or
   class member variables appearing after the `.` or `->` operators highlighted,
   use for example [vim-cpp-modern][vim-cpp-modern].

3. Functions are not highlighted by default. There are plenty of other Vim
   syntax plugins that provide this feature for C/C++ files. See for example
   [vim-cpp-modern][vim-cpp-modern].


## Optional features

```vim
" Enable highlighting of CUDA kernel calls
let g:cuda_kernel_highlight = 1

" Highlight keywords from CUDA Runtime API
let g:cuda_runtime_api_highlight = 1

" Highlight keywords from CUDA Driver API
let g:cuda_driver_api_highlight = 1

" Highlight keywords from CUDA Thrust library
let g:cuda_thrust_highlight = 1
```


## Installation

```bash
$ cd ~/.vim/pack/git-plugins/start
$ git clone https://github.com/bfrg/vim-cuda-syntax
```
**Note:** The directory name `git-plugins` is arbitrary, you can pick any other
name. For more details see `:help packages`.

Alternatively, use your favorite plugin manager.


## FAQ

> I want everything in-between the triple angle-brackets highlighted to get a
> more distinct highlighting of kernel calls.

Add the following lines to `~/.vim/after/syntax/cuda.vim` (create the file if
it doesn't exist):
```vim
syntax match cudaKernelAngles "<<<\_.\{-}>>>"
highlight link cudaKernelAngles Operator
```

> I want the CUDA language extensions (`__device__`, `__host__`, etc.)
> highlighted like the standard C/C++ keywords.

Add the following to `~/.vim/after/syntax/cuda.vim`:
```vim
highlight link cudaStorageClass Statement
```


## License

Distributed under the same terms as Vim itself. See `:help license`.


[toolkit]: https://docs.nvidia.com/cuda/index.html
[data-fields]: https://docs.nvidia.com/cuda/cuda-runtime-api/functions.html#functions
[vim-cpp-modern]: https://github.com/bfrg/vim-cpp-modern

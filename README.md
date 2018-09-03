# vim-cuda-syntax

This file provides syntax highlighting for CUDA development in Vim. Compared to
Vim's default CUDA syntax file, it adds highlighting of all CUDA defined:
- types
- enums
- constants
- global variables
- functions (see details below)
- namespaces
- thrust keywords

as well as highlighting of the triple-angle brackets in CUDA kernel calls.

All keywords were accumulated from the
[CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html).

#### Limitations

1. Not all CUDA library functions are highlighted by default. To get
   highlighting of function names either use one of the popular C/C++ syntax
   plugins ([vim-cpp-enhanced-highlight](https://github.com/octol/vim-cpp-enhanced-highlight)
   or [vim-cpp-modern](https://github.com/bfrg/vim-cpp-modern)), or add the
   following lines to `~/.vim/after/c.vim`:
   ```vim
   syntax match cUserFunction "\<\h\w*\>\(\s\|\n\)*("me=e-1 contains=cParen,cCppParen
   highlight default link cUserFunction Function
   ```
   This will highlight all words followed by an opening parenthesis as
   `Function`.

   This simple regex, however, doesn't work for function templates that are
   called with template arguments, like `foo<32, bar<123>>(xyz)`. Matching the
   function name `foo` in such calls isn't trivial. Therefore, we explicitly
   added all CUDA function templates so that they always get highlighted, even
   when called with template arguments.

2. Highlighting of the triple angle-brackets in CUDA kernel calls works only
   when the angle brackets are on the same line. The function name is only
   highlighted when called without template arguments, i.e. `mykernel` won't be
   highlighted in `mykernel<foo, bar><<<grid, threads>>>(data)`, for the same
   reason as above.

3. CUDA [data fields](https://docs.nvidia.com/cuda/cuda-runtime-api/functions.html#functions)
   are not highlighted because many keywords have familiar names which could
   collide with either user-defined variables (like `ptr`, `x`, `y`), or with
   C++ standard library types (like `function` or `array`) and would mess up the
   highlighting.


## Optional features

```vim
" Highlight keywords from CUDA Runtime API
let g:cuda_runtime_api_highlight = 1

" Highlight keywords from CUDA Driver API
let g:cuda_driver_api_highlight = 1

" Highlight keywords from CUDA Thrust library
let g:cuda_thrust_highlight = 1

" Disable highlighting of CUDA kernel calls
let g:cuda_no_kernel_highlight = 1
```


## Installation

Since this syntax file fully replaces Vim's default CUDA syntax file, copy the
`cuda.vim` file into the `~/.vim/syntax` directory.


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

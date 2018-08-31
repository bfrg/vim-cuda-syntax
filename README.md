# vim-cuda-syntax: Enhanced CUDA syntax highlighting

This syntax file provides enhanced CUDA syntax highlighting for Vim, including
highlighting of CUDA kernel calls.

## Additions to Vim's default CUDA syntax file

Highlighting of all CUDA defined
- Types
- Enums
- Constants
- Global variables
- Library functions (see details below)
- Namespaces
- Thrust keywords
- triple-angle brackets in CUDA Kernel calls

All keywords were accumulated from the
[CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html).

#### Limitations

1. CUDA library functions are not highlighted. To get highlighting of function
   names either use one of the popular C/C++ syntax plugins
   ([vim-cpp-enhanced-highlight](https://github.com/octol/vim-cpp-enhanced-highlight)
   or [vim-cpp-modern](https://github.com/bfrg/vim-cpp-modern)), or add the
   following snippet to `~/.vim/after/c.vim`:
   ```vim
   syntax match cUserFunction "\<\h\w*\>\(\s\|\n\)*("me=e-1 contains=cParen,cCppParen
   highlight default link cUserFunction Function
   ```
   This will highlight all words that are followed by an opening parenthesis as
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


## License

Distributed under the same terms as Vim itself. See `:help license`.

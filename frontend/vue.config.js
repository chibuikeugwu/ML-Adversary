const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave:false,
  devServer:{
    proxy: {
        '/api': {
          target: 'https://api.openai.com',
          changeOrigin: true,
        }
      }
  },
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.wasm$/,
          type: 'javascript/auto',
          loader: 'wasm-loader',
          options: {
            name: 'wasm/[name].[hash:8].[ext]',
          },
        },

      ],
    },
  },
  // configureWebpack: {
  //   configure: (config) => {
  //     config.experiments = {
  //       asyncWebAssembly: true,
  //       layers: true,
  //     };
  //     // turn off static file serving of WASM files
  //     // we need to let Webpack handle WASM import
  //     config.module.rules
  //         .find((i) => "oneOf" in i)
  //         .oneOf.find((i) => i.type === "asset/resource")
  //         .exclude.push(/\.wasm$/);
  //     return config;
  //   },
  // },

})

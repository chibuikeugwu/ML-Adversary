import Vue from 'vue'
import App from './App.vue'

Vue.config.productionTip = false

import store from './store'
import router from './router/index'
import ElementUI from 'element-ui'
import VueRouter from 'vue-router'
import 'element-ui/lib/theme-chalk/index.css';
import './assets/css/main.css'

Vue.use(VueRouter)
Vue.use(ElementUI)

new Vue({
  el:'#app',
  store,
  router,
  render: h => h(App),
  beforeCreate() {
    Vue.prototype.$bus=this
  }
})

<template>
  <div>
    <TheMenu/>
    <TheHeader/>
    <div class="content-box" :class="{'content-collapse':collapse}">
      <router-view></router-view>
    </div>

  </div>

</template>

<script>

import TheHeader from "@/components/TheHeader";
import TheMenu from "@/components/TheMenu";

export default {
  name:'Home',
  data(){
    return {
      collapse:false
    }
  },
  components:{
    TheHeader,TheMenu
  },
  mounted() {
    this.$bus.$on('menuCollapse',()=>{
      this.collapse = !this.collapse})
  },
  beforeDestroy() {
    this.$bus.$off('menuCollapse')
  }
}
</script>

<style>

</style>
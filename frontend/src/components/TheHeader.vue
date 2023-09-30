<template>
  <div class="header">
    <div class="collapse-btn">
      <i class="el-icon-menu" @click="foldMenu"></i>
    </div>
    <img src="../assets/img/logo-plaintext.png"/>
    <el-input class="searchInput"
        placeholder="Search"
        suffix-icon="el-icon-search"
        v-model="searchContent"
        @keydown.native.enter="search">
    </el-input>
  </div>

</template>

<script>


export default {
  name:'TheHeader',
  props:['userName'],
  data(){
    return {
      searchContent:''
    }
  },
  methods:{
    foldMenu(){
      this.$bus.$emit("menuCollapse",null)
    },
    search(){
      this.$router.replace({
        name:'SearchSet',
        params:{searchKey:this.searchContent}
      })
    }
  },
  // computed:{
  //   currentUserName(){
  //     let currentUser=JSON.parse(window.sessionStorage.getItem('user'))
  //     return currentUser.userName
  //   }
  // }
}
</script>

<style scoped>

.header {
  position: relative;
  background-color: #903749;
  box-sizing: border-box;
  width: 100%;
  height: 70px;
  font-size: 22px;
  color: white;
  display: flex;
}

.collapse-btn{
  float: left;
  padding: 0 21px;
  cursor: pointer;
  line-height: 70px;
}


img {
  height: 20px;
  position: relative;
  float: left;
  display: flex;
  top: 25px;
}

.searchInput {
  margin-top: auto;
  margin-bottom: auto;
  width: 400px;
  left: 300px;
}

/*.welcomeText {*/
/*  position: relative;*/
/*  float: left;*/
/*  display: flex;*/
/*  color: white;*/
/*  margin-top: auto;*/
/*  margin-bottom: auto;*/
/*  left: 600px;*/
/*  font-size: 20px;*/
/*}*/


</style>
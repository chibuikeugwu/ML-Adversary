<template>
  <div class="sideBar">
        <el-menu
            class="el-menu-vertical-demo"
            :default-active="onRoutes"
            :collapse="collapse"
            background-color="#903749"
            text-color="#fff"
            active-text-color="#ffd04b">
<!--            router>-->

          <el-menu-item class="avatarMenuItem" v-if="!collapse">
            <div class="avatar">
              <el-avatar src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"></el-avatar>
            </div>
          </el-menu-item>

          <el-menu-item index="HomeInfo" @click="switchMenu($event)">
            <i class="el-icon-house"></i>
            <span slot="title">Home</span>
          </el-menu-item>

          <el-menu-item index="Pictures" @click="switchMenu($event)">
            <i class="el-icon-collection"></i>
            <span slot="title">Pictures</span>
          </el-menu-item>

          <el-menu-item index="Statistics" @click="switchMenu($event)">
            <i class="el-icon-s-data"></i>
            <span slot="title">Statistics</span>
          </el-menu-item>

          <el-menu-item class="MenuDivider">
            <el-divider></el-divider>
          </el-menu-item>

          <el-menu-item>
            <i class="el-icon-turn-off"></i>
            <span slot="title">Theme</span>
          </el-menu-item>

          <el-menu-item index="UserSettings" @click="switchMenu($event)">
            <i class="el-icon-setting"></i>
            <span slot="title">Settings</span>
          </el-menu-item>

          <el-menu-item index="Help" @click="switchMenu($event)">
            <i class="el-icon-info"></i>
            <span slot="title">Help</span>
          </el-menu-item>

          <el-menu-item @click="logOut">
            <i class="el-icon-switch-button"></i>
            <span slot="title">Exit</span>
          </el-menu-item>

        </el-menu>
  </div>
</template>

<script>


export default {
  name:'TheMenu',
  data(){
    return{
      collapse:false
    }
  },
  methods:{
    switchMenu(e){
      this.$router.replace({name:e.index})
    },
    logOut(){
      this.$confirm('确定退出吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        window.sessionStorage.removeItem('user')
        this.$router.replace('/')
        this.$message({
          type: 'success',
          message: '已退出MemoryMaster!'
        });
      }).catch(() => {});
    }
  },
  computed:{
      onRoutes(){
        return this.$route.path.replace('/','')
      }
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

<style scoped>

.sideBar {
  display: block;
  position: absolute;
  left: 0;
  top:70px;
  bottom: 0;
  background-color: #903749;
  font-weight: bolder;
}


.sideBar > ul {
  height: 100%;
}

.avatarMenuItem {
  display: flex;
}

.avatar {
  margin: auto;
}

el-divider {
  margin: auto;
}

.MenuDivider {
  display: flex;
}

</style>